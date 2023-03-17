from dataclasses import dataclass
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
    cross_entropy,
)
from torch import Tensor
from typing import Optional
import torch
import warnings


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
        num_classes: int = 1,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_dim, num_classes, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        l2_penalty: float = 1.0,
        max_iter: int = 10_000,
        tol: float = 1e-4,
    ) -> float:
        """Fits the model to the input data using L-BFGS with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            l2_penalty: L2 regularization strength.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.
            tol: Tolerance for the L-BFGS optimizer.

        Returns:
            Final value of the loss function after optimization.
        """

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_grad=tol,
        )

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        loss = torch.inf
        y = y.float()

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # Calculate the loss function
            logits = self(x).squeeze(-1)
            loss = loss_fn(logits, y)

            # Add L2 regularization penalty the way scikit-learn does
            l2_reg = 0.5 * self.linear.weight.square().sum()

            reg_loss = loss + l2_penalty * l2_reg
            reg_loss.backward()

            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def fit_cv(
        self,
        x: Tensor,
        y: Tensor,
        k: int = 5,
        *,
        max_iter: int = 10_000,
        num_penalties: int = 10,
        seed: int = 42,
        tol: float = 1e-4,
    ) -> RegularizationPath:
        """Fit using k-fold cross-validation to select the best L2 penalty.

        Args:
            model: Instance of the Classifier class.
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            k: Number of folds for k-fold cross-validation.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.
            num_penalties: Number of L2 regularization penalties to try.
            seed: Random seed for the k-fold cross-validation.
            tol: Tolerance for the L-BFGS optimizer.

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

        l2_penalties = torch.logspace(-4, -4, num_penalties).tolist()
        y = y.float()

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        losses = x.new_zeros((k, num_penalties))

        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            train_indices = torch.cat([indices[:start], indices[end:]])
            val_indices = indices[start:end]

            train_x, train_y = x[train_indices], y[train_indices]
            val_x, val_y = x[val_indices], y[val_indices]

            # Regularization path with warm-starting
            for j, l2_penalty in enumerate(l2_penalties):
                self.fit(
                    train_x, train_y, l2_penalty=l2_penalty, max_iter=max_iter, tol=tol
                )

                logits = self(val_x).squeeze(-1)
                loss = loss_fn(logits, val_y)
                losses[i, j] = loss

        mean_losses = losses.mean(dim=0)
        best_idx = mean_losses.argmin()

        # Check the health of the regularization path- we expect a U shape
        lhs, rhs = mean_losses[:best_idx], mean_losses[best_idx:]
        if not len(lhs) or not len(rhs):
            warnings.warn("The best penalty is at the edge of the regularization path.")
        else:
            lhs_monotonic_decreasing = lhs.diff().le(0).all()
            rhs_monotonic_increasing = rhs.diff().ge(0).all()

            if not lhs_monotonic_decreasing or not rhs_monotonic_increasing:
                warnings.warn("The regularization path does not have a U shape.")

        # Refit with the best penalty
        best_penalty = l2_penalties[best_idx]
        self.fit(x, y, l2_penalty=best_penalty, max_iter=max_iter, tol=tol)
        return RegularizationPath(l2_penalties, mean_losses.tolist())

    def nullspace_project(self, x: Tensor) -> Tensor:
        """Project the given data onto the nullspace of the classifier."""

        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        A = self.linear.weight.data.T
        P = A @ torch.linalg.solve(A.mT @ A, A.mT)
        return x - P @ x
