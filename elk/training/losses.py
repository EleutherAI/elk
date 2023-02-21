"""Loss functions for training reporters."""

import math

import torch
from torch import Tensor


def H(p: Tensor) -> Tensor:
    """Entropy of Bernoulli distribution(s) with success probability `p`."""
    return torch.nn.functional.binary_cross_entropy(p, p)


# See to have the same exact type signature as ccs_squared_loss,
# even if we have defaults
# see https://github.com/python/mypy/issues/10740#issuecomment-878622464
def js_loss(
    logit0: Tensor,
    logit1: Tensor,
) -> Tensor:
    """Consistency and confidence loss based on the Jensen-Shannon divergence.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1]."""
    return _js_loss(logit0, logit1, confidence=0.0, base=2.0)


def _js_loss(
    logit0: Tensor,
    logit1: Tensor,
    confidence: float,
    base: float,
) -> Tensor:
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
