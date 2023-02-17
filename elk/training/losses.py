"""Loss functions for training reporters."""

from torch import Tensor
import math
import torch


def H(p: Tensor) -> Tensor:
    """Entropy of Bernoulli distribution(s) with success probability `p`."""
    return torch.nn.functional.binary_cross_entropy(p, p)


def js_loss(
    logit0: Tensor,
    logit1: Tensor,
    confidence: float = 0.0,
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
