from typing import NamedTuple

import torch
from torch import Tensor


def to_one_hot(labels: Tensor, n_classes: int) -> Tensor:
    """
    Convert a tensor of class labels to a one-hot representation.

    Args:
        labels (Tensor): A tensor of class labels of shape (N,).
        n_classes (int): The total number of unique classes.

    Returns:
        Tensor: A one-hot representation tensor of shape (N, n_classes).
    """
    one_hot_labels = labels.new_zeros(labels.size(0), n_classes)
    return one_hot_labels.scatter_(1, labels.unsqueeze(1).long(), 1)


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Compute the accuracy of a classification model.

    Args:
        y_true: Ground truth tensor of shape (N,).
        y_pred: Predicted class tensor of shape (N,) or (N, n_classes).

    Returns:
        float: Accuracy of the model.
    """
    # Check if binary or multi-class classification
    if len(y_pred.shape) == 1:
        hard_preds = y_pred > 0.5
    else:
        hard_preds = y_pred.argmax(-1)

    return hard_preds.cpu().eq(y_true.cpu()).float().mean().item()


class RocAucResult(NamedTuple):
    """Named tuple for storing ROC AUC results."""

    estimate: float
    """Point estimate of the ROC AUC computed on this sample."""
    lower: float
    """Lower bound of the bootstrap confidence interval."""
    upper: float
    """Upper bound of the bootstrap confidence interval."""


def roc_auc(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Area under the receiver operating characteristic curve (ROC AUC).

    Unlike scikit-learn's implementation, this function supports batched inputs of
    shape `(N, n)` where `N` is the number of datasets and `n` is the number of samples
    within each dataset. This is primarily useful for efficiently computing bootstrap
    confidence intervals.

    Args:
        y_true: Ground truth tensor of shape `(N,)` or `(N, n)`.
        y_pred: Predicted class tensor of shape `(N,)` or `(N, n)`.

    Returns:
        Tensor: If the inputs are 1D, a scalar containing the ROC AUC. If they're 2D,
            a tensor of shape (N,) containing the ROC AUC for each dataset.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred should have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.dim() not in (1, 2):
        raise ValueError("y_true and y_pred should be 1D or 2D tensors")

    # Sort y_pred in descending order and get indices
    indices = y_pred.argsort(descending=True, dim=-1)

    # Reorder y_true based on sorted y_pred indices
    y_true_sorted = y_true.gather(-1, indices)

    # Calculate number of positive and negative samples
    num_positives = y_true.sum(dim=-1)
    num_negatives = y_true.shape[-1] - num_positives

    # Calculate cumulative sum of true positive counts (TPs)
    tps = torch.cumsum(y_true_sorted, dim=-1)

    # Calculate cumulative sum of false positive counts (FPs)
    fps = torch.cumsum(1 - y_true_sorted, dim=-1)

    # Calculate true positive rate (TPR) and false positive rate (FPR)
    tpr = tps / num_positives.view(-1, 1)
    fpr = fps / num_negatives.view(-1, 1)

    # Calculate differences between consecutive FPR values (widths of trapezoids)
    fpr_diffs = torch.cat(
        [fpr[..., 1:] - fpr[..., :-1], torch.zeros_like(fpr[..., :1])], dim=-1
    )

    # Calculate area under the ROC curve for each dataset using trapezoidal rule
    return torch.sum(tpr * fpr_diffs, dim=-1).squeeze()


def roc_auc_ci(
    y_true: Tensor,
    y_pred: Tensor,
    *,
    num_samples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> RocAucResult:
    """Bootstrap confidence interval for the ROC AUC.

    Args:
        y_true: Ground truth tensor of shape `(N,)`.
        y_pred: Predicted class tensor of shape `(N,)`.
        num_samples (int): Number of bootstrap samples to use.
        level (float): Confidence level of the confidence interval.
        seed (int): Random seed for reproducibility.

    Returns:
        RocAucResult: Named tuple containing the lower and upper bounds of the
            confidence interval, along with the point estimate.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred should have the same shape; "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.dim() != 1:
        raise ValueError("y_true and y_pred should be 1D tensors")

    device = y_true.device
    N = y_true.shape[0]

    # Generate random indices for bootstrap samples (shape: [num_bootstraps, N])
    rng = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randint(0, N, (num_samples, N), device=device, generator=rng)

    # Create bootstrap samples of true labels and predicted probabilities
    y_true_bootstraps = y_true[indices]
    y_pred_bootstraps = y_pred[indices]

    # Compute ROC AUC scores for bootstrap samples
    bootstrap_aucs = roc_auc(y_true_bootstraps, y_pred_bootstraps)

    # Calculate the lower and upper bounds of the confidence interval. We use
    # nanquantile instead of quantile because some bootstrap samples may have
    # NaN values due to the fact that they have only one class.
    alpha = (1 - level) / 2
    q = y_pred.new_tensor([alpha, 1 - alpha])
    lower, upper = bootstrap_aucs.nanquantile(q).tolist()

    # Compute the point estimate
    estimate = roc_auc(y_true, y_pred).item()
    return RocAucResult(estimate, lower, upper)
