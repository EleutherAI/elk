"""Loss functions for training models."""

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

    Args:
        logit0: The logits for the first model.
        logit1: The logits for the second model.

    Returns:
        tuple containing the consistency and confidence loss.
        The consistency is the squared difference between the probabilities.
        The confidence is the squared minimum of the probabilities.
    """
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()

    consistency = p0.sub(1 - p1).square().mean()
    confidence = torch.min(p0, p1).square().mean()

    return consistency + confidence
