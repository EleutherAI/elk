from dataclasses import dataclass, field

import torch
from concept_erasure import LeaceEraser
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
)
from torch.nn.functional import (
    cross_entropy,
)


@dataclass
class InlpResult:
    """Result of Iterative Nullspace Projection (NLP)."""

    losses: list[float] = field(default_factory=list)
    classifiers: list["Classifier"] = field(default_factory=list)


@dataclass
class RegularizationPath:
    """Result of cross-validation."""

    penalties: list[float]
    losses: list[float]

    @property
    def best_penalty(self) -> float:
        """Returns the best L2 regularization penalty."""
        return self.penalties[self.losses.index(self.best_loss)]

    @property
    def best_loss(self) -> float:
        """Returns the best loss."""
        return min(self.losses)


class Classifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        eraser: LeaceEraser | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_dim, num_classes if num_classes > 2 else 1, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()
        self.eraser = eraser

    def forward(self, x: Tensor) -> Tensor:
        if self.eraser is not None:
            x = self.eraser(x)
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        alpha: float = 0.001,
        l1_ratio: float = 0,
        max_iter: int = 10_000,
    ) -> float:
        """Fits the model to the input data using L-BFGS with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            alpha: L2 regularization strength.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.

        Returns:
            Final value of the loss function after optimization.
        """
        # Use cuML backend for LASSO
        if l1_ratio > 0:
            from cuml import LogisticRegression

            model = LogisticRegression(
                C=1 / alpha, penalty="elasticnet", l1_ratio=l1_ratio
            )
            model.fit(x, y)

            W = torch.as_tensor(model.coef_).unsqueeze(0).to(x.device)
            b = torch.as_tensor(model.intercept_).to(x.device)

            self.linear.weight.data = W
            self.linear.bias.data = b
            return 0.0

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        loss = torch.inf
        y = y.to(
            torch.get_default_dtype() if num_classes == 1 else torch.long,
        )

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # Calculate the loss function
            logits = self(x).squeeze(-1)
            loss = loss_fn(logits, y)
            if alpha:
                reg_loss = loss + alpha * self.linear.weight.square().sum()
            else:
                reg_loss = loss

            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def fit_cv(
        self,
        x: Tensor,
        y: Tensor,
        *,
        k: int = 5,
        l1_ratio: float = 0.0,
        max_iter: int = 10_000,
        num_penalties: int = 10,
        seed: int = 42,
    ) -> RegularizationPath:
        """Fit using k-fold cross-validation to select the best L2 penalty.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            k: Number of folds for k-fold cross-validation.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.
            num_penalties: Number of L2 regularization penalties to try.
            seed: Random seed for the k-fold cross-validation.

        Returns:
            `RegularizationPath` containing the penalties tried and the validation loss
            achieved using that penalty, averaged across the folds.
        """
        num_samples = x.shape[0]
        if k < 3:
            raise ValueError("`k` must be at least 3")
        if k > num_samples:
            raise ValueError("`k` must be less than or equal to the number of samples")

        rng = torch.Generator(device=x.device)
        rng.manual_seed(seed)

        fold_size = num_samples // k
        indices = torch.randperm(num_samples, device=x.device, generator=rng)

        # Try a range of L2 penalties, including 0
        penalties = [0.0] + torch.logspace(-4, 4, num_penalties).tolist()

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        losses = x.new_zeros((k, num_penalties + 1))
        y = y.to(
            torch.get_default_dtype() if num_classes == 1 else torch.long,
        )

        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            train_indices = torch.cat([indices[:start], indices[end:]])
            val_indices = indices[start:end]

            train_x, train_y = x[train_indices], y[train_indices]
            val_x, val_y = x[val_indices], y[val_indices]

            # Regularization path with warm-starting
            for j, penalty in enumerate(penalties):
                self.fit(
                    train_x,
                    train_y,
                    alpha=penalty,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                )

                logits = self(val_x).squeeze(-1)
                loss = loss_fn(logits, val_y)
                losses[i, j] = loss

        mean_losses = losses.mean(dim=0)
        best_idx = mean_losses.argmin()

        # Refit with the best penalty
        best_penalty = penalties[best_idx]
        self.fit(x, y, alpha=best_penalty, l1_ratio=l1_ratio, max_iter=max_iter)
        return RegularizationPath(penalties, mean_losses.tolist())

    @classmethod
    def inlp(
        cls,
        x: Tensor,
        y: Tensor,
        eraser: LeaceEraser | None = None,
        l1_ratio: float = 0.0,
        max_iter: int | None = None,
        tol: float = 0.01,
    ) -> InlpResult:
        """Iterative Nullspace Projection (INLP) <https://arxiv.org/abs/2004.07667>.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            eraser: Concept erasure function to use. If `None`, no erasure is performed.
            max_iter: Maximum number of iterations to run. If `None`, run until the data
                is linearly guarded (no linear classifier can extract information)
            tol: Tolerance for the loss function. The algorithm will stop when the loss
                is within `tol` of the entropy of the labels.

        Returns:
            `InlpResult` containing the classifiers and losses achieved at each
            iteration.
        """

        y.shape[-1] if y.ndim > 1 else 2
        d = x.shape[-1]
        loss = 0.0

        # Compute entropy of the labels
        p = y.float().mean()
        H = -p * torch.log(p) - (1 - p) * torch.log(1 - p)

        max_iter = max_iter or d

        # Iterate until the loss is within epsilon of the entropy
        # meaning LR is not able to find a useful classifier anymore
        result = InlpResult()
        for _ in range(max_iter):
            clf = cls(d, eraser=eraser, device=x.device, dtype=x.dtype)
            loss = clf.fit(x, y)
            result.classifiers.append(clf)
            result.losses.append(loss)

            if loss >= (1.0 - tol) * H:
                break

            # Project the data onto the nullspace of the classifier
            x = clf.nullspace_project(x)

        return result

    def nullspace_project(self, x: Tensor) -> Tensor:
        """Project the given data onto the nullspace of the classifier."""

        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        A = self.linear.weight.data.T
        P = A @ torch.linalg.solve(A.mT @ A, A.mT)
        return x - x @ P
