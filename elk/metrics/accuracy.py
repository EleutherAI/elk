from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions.normal import Normal


@dataclass(frozen=True)
class AccuracyResult:
    """Accuracy point estimate and confidence interval."""

    estimate: float
    """Point estimate of the accuracy computed on this sample."""
    lower: float
    """Lower bound of the confidence interval."""
    upper: float
    """Upper bound of the confidence interval."""


def accuracy_ci(
    y_true: Tensor, y_pred: Tensor, *, level: float = 0.95
) -> AccuracyResult:
    """
    Compute the accuracy of a classifier and its confidence interval.

    Args:
        y_true: Ground truth tensor of shape (N,).
        y_pred: Predicted class tensor of shape (N,).

    Returns:
        float: Accuracy of the model.
    """
    # We expect the inputs to be integers
    assert not torch.is_floating_point(y_pred) and not torch.is_floating_point(y_true)
    assert y_true.shape == y_pred.shape

    # Point estimate of the accuracy
    acc = y_pred.eq(y_true).float().mean()

    # Compute the CI quantiles
    alpha = (1 - level) / 2
    q = acc.new_tensor([alpha, 1 - alpha])

    # Normal approximation to the binomial distribution
    stderr = (acc * (1 - acc) / len(y_true)) ** 0.5
    lower, upper = Normal(acc, stderr).icdf(q).tolist()

    return AccuracyResult(acc.item(), lower, upper)
