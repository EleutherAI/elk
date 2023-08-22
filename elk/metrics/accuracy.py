from dataclasses import dataclass
from typing import Literal
from einops import repeat

import torch
from torch import Tensor


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
    y_true: Tensor,
    y_logits: Tensor,
    ensembling: Literal["none", "partial", "full", "old"] = "none",
    *,
    num_samples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> AccuracyResult:
    """Bootstrap confidence interval for accuracy, with optional clustering.

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

    THRESHOLD = 0.5  # after platt_scaling
    # print("ensembling", ensembling)
    def pool_logits(logits):
        return (logits[..., 1] - (logits[..., 0] - 1)) / 2
    if ensembling == "full":
        # y_logits [1000, 13, 2] becomes [1000]
        y_logits_pooled = pool_logits(y_logits)
        y_pred = y_logits_pooled.mean(dim=-1).gt(THRESHOLD)
        y_true_rep = y_true # y_true stays the same
    elif ensembling == "partial":
        # y_logits [1000, 13, 2] becomes [1000, 13]
        v = y_logits.shape[-2]
        y_true_rep = repeat(y_true, "n -> (n v)", v=v)
        y_logits_pooled = pool_logits(y_logits)
        y_pred = y_logits_pooled.flatten().gt(THRESHOLD)
    elif ensembling == "none":
        # y_logits [1000, 13, 2] but uhhh... we take just the positive?
        v, c = y_logits.shape[-2:]
        y_true_rep = repeat(y_true, "n -> (n v)", v=v)
        y_pred = y_logits[..., 1].flatten().gt(THRESHOLD)
    elif ensembling == "old":
        rest_vals = y_logits.shape[1:-1] # skipping first (n) and last (c)
        rest_labels = " ".join([f"v{i}" for i in range(len(rest_vals))])
        rest = {f"v{i}": v for i, v in enumerate(rest_vals)}
        # print(rest_vals)
        # print(rest_labels)
        # print(rest)
        y_true_rep = repeat(y_true, f"n -> n {rest_labels}", **rest)
        y_pred = y_logits.argmax(dim=-1)
    # print("y_true", y_true_rep.shape)
    # print("y_pred", y_pred.shape)

    if any([torch.is_floating_point(t) for t in (y_pred, y_true_rep)]):
        raise TypeError("y_true and y_pred should be integer tensors")
    if y_true_rep.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred should have the same shape; "
            f"got {y_true_rep.shape} and {y_pred.shape}"
        )
    if y_true_rep.dim() not in (1, 2):
        raise ValueError("y_true and y_pred should be 1D or 2D tensors")
    # Either the number of samples (1D) or the number of clusters (2D)
    N = y_true_rep.shape[0]
    device = y_true_rep.device

    # Generate random indices for bootstrap samples (shape: [num_bootstraps, N])
    rng = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randint(0, N, (num_samples, N), device=device, generator=rng)

    # Create bootstrap samples of true labels and predicted probabilities
    y_true_bootstraps = y_true_rep[indices]
    y_pred_bootstraps = y_pred[indices]

    # Compute ROC AUC scores for bootstrap samples. If the inputs were 2D, the
    # bootstrapped tensors are now 3D [num_bootstraps, N, cluster_size], so we
    # call flatten(1) to get a 2D tensor [num_bootstraps, N * cluster_size].
    bootstrap_hits = y_true_bootstraps.flatten(1).eq(y_pred_bootstraps.flatten(1))
    bootstrap_accs = bootstrap_hits.float().mean(1)

    # Calculate the lower and upper bounds of the confidence interval. We use
    # nanquantile instead of quantile because some bootstrap samples may have
    # NaN values due to the fact that they have only one class.
    alpha = (1 - level) / 2
    q = bootstrap_accs.new_tensor([alpha, 1 - alpha])
    lower, upper = bootstrap_accs.nanquantile(q).tolist()

    # Compute the point estimate. Call flatten to ensure that we get a single number
    # computed across cluster boundaries even if the inputs were clustered.
    estimate = y_true_rep.flatten().eq(y_pred.flatten()).float().mean().item()
    return AccuracyResult(estimate, lower, upper)
