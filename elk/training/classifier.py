from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
    cross_entropy,
    sigmoid,
)
from typing import Optional
import torch


class Classifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self, input_dim: int, num_classes: int = 1, device: Optional[str] = None
    ):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, num_classes, device=device)
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        max_iter: int = 10_000,
    ) -> float:
        """Fit parameters to the given data with LBFGS."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            # Tolerance used by scikit-learn's LogisticRegression
            tolerance_grad=1e-4,
        )

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        loss = torch.inf
        y = y.float()

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            loss = loss_fn(self(x).squeeze(-1), y)
            loss.backward()

            return float(loss)

        optimizer.step(closure)
        return float(loss)

    def nullspace_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project the given data onto the nullspace of the classifier."""

        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        A = self.linear.weight.data.T
        P = A @ torch.linalg.solve(A.mT @ A, A.mT)
        return x - P @ x
