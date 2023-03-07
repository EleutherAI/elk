"""Loss functions for training reporters."""

from torch import Tensor
import math
import torch
import warnings


def H(p: Tensor) -> Tensor:
    """Entropy of Bernoulli distribution(s) with success probability `p`."""
    return torch.nn.functional.binary_cross_entropy(p, p)


def js_loss(
    logit0: Tensor,
    logit1: Tensor,
    confidence: float = 0.0,  # TODO: If it's too large we get nans
    base: float = 2.0,
) -> Tensor:
    """Consistency and confidence loss based on the Jensen-Shannon divergence.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1]."""
    p0, neg_p1 = logit0.sigmoid(), 1 - logit1.sigmoid()
    nats = (1 + confidence) * H((p0 + neg_p1) / 2) - (H(p0) + H(neg_p1)) / 2
    return nats / math.log(base)


def ccs_squared_loss(logit0: Tensor, logit1: Tensor) -> Tensor:
    """CCS loss from original paper, with squared differences between probabilities.

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition.
        logit1: The log odds for the negated proposition.

    Returns:
        The sum of the consistency and confidence losses.
    """
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()
    consistency = p0.sub(1 - p1).square().mean()
    confidence = torch.min(p0, p1).square().mean()

    return consistency + confidence


def prompt_var_loss(logit0: Tensor, logit1: Tensor) -> Tensor:
    """
    The prompt-variance CCS loss.
    This is the original CCS loss with an additional term: the squared
    difference between the probability of a proposition and the mean probability
    over all variants of that proposition (templates).

    The loss is symmetric, so it doesn't matter which argument is the original and
    which is the negated proposition.

    Args:
        logit0: The log odds for the original proposition. shape ([batch,] n_variants)
        logit1: The log odds for the negated proposition. shape ([batch,] n_variants)

    Returns:
        The sum of the negation consistency, confidence, and prompt invariance losses.

    """
    assert logit0.shape == logit1.shape
    assert len(logit0.shape) in [1, 2]
    if logit0.shape[-1] == 1:
        warnings.warn(
            "Only one variant provided. Prompt variance loss will equal CCS loss."
        )
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()
    consistency = p0.sub(1 - p1).square().mean()
    confidence = torch.min(p0, p1).square().mean()
    mean_p0, mean_p1 = p0.mean(dim=-1, keepdim=True), p1.mean(dim=-1, keepdim=True)
    prompt_variance = (mean_p0 - p0).square().mean() + (mean_p1 - p1).square().mean()
    return consistency + confidence + prompt_variance
