from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
    cross_entropy,
)
from torch import Tensor
from typing import Optional
import torch


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
            l2_reg = 0.5 * self.linear.weight.data.square().sum()

            # scikit-learn sums the loss over the number of samples, so this is
            # equivalent to the L2 regularization penalty being divided by # samples
            loss += l2_penalty * l2_reg / x.shape[0]
            loss.backward()

            return float(loss)

        for _ in range(1):
            optimizer.step(closure)
        return float(loss)

    def nullspace_project(self, x: Tensor) -> Tensor:
        """Project the given data onto the nullspace of the classifier."""

        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        A = self.linear.weight.data.T
        P = A @ torch.linalg.solve(A.mT @ A, A.mT)
        return x - P @ x
