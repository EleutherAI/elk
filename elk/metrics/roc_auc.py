from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RocAucResult:
    """Dataclass for storing ROC AUC results."""

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
    """Bootstrap confidence interval for the ROC AUC, with optional clustering.

    When the input arguments are 2D, this function performs the cluster bootstrap,
    resampling clusters with replacement instead of individual samples. The first
    axis is assumed to be the cluster axis.

    Args:
        y_true: Ground truth tensor of shape `(N,)` or `(N, cluster_size)`.
        y_pred: Predicted class tensor of shape `(N,)` or `(N, cluster_size)`.
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
    if y_true.dim() not in (1, 2):
        raise ValueError("y_true and y_pred should be 1D or 2D tensors")

    # Either the number of samples (1D) or the number of clusters (2D)
    N = y_true.shape[0]
    device = y_true.device

    # Generate random indices for bootstrap samples (shape: [num_bootstraps, N])
    rng = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randint(0, N, (num_samples, N), device=device, generator=rng)

    # Create bootstrap samples of true labels and predicted probabilities
    y_true_bootstraps = y_true[indices]
    y_pred_bootstraps = y_pred[indices]

    # Compute ROC AUC scores for bootstrap samples. If the inputs were 2D, the
    # bootstrapped tensors are now 3D [num_bootstraps, N, cluster_size], so we
    # call flatten(1) to get a 2D tensor [num_bootstraps, N * cluster_size].
    bootstrap_aucs = roc_auc(y_true_bootstraps.flatten(1), y_pred_bootstraps.flatten(1))

    # Calculate the lower and upper bounds of the confidence interval. We use
    # nanquantile instead of quantile because some bootstrap samples may have
    # NaN values due to the fact that they have only one class.
    alpha = (1 - level) / 2
    q = bootstrap_aucs.new_tensor([alpha, 1 - alpha])
    lower, upper = bootstrap_aucs.nanquantile(q).tolist()

    # Compute the point estimate. Call flatten to ensure that we get a single number
    # computed across cluster boundaries even if the inputs were clustered.
    estimate = roc_auc(y_true.flatten(), y_pred.flatten()).item()
    return RocAucResult(estimate, lower, upper)
