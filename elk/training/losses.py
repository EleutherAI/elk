from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
import math
import torch


def bernoulli_js(
    logit0: torch.Tensor, logit1: torch.Tensor, base: float = 2.0, alpha: float = 1.0
):
    """Jensen-Shannon divergence between Bernoulli distributions.

    Note that by default we use the base 2 logarithm, so the value is measured in bits.
    This ensures the divergence is in the range [0, 1].
    """
    # Arithmetic mixture of the two distributions. For numerical stability, we
    # do the operation in log space.
    log_p_bar = torch.stack([logit0, logit1]).logsumexp(0) - math.log(2)

    H_bar = bce_logits(log_p_bar, log_p_bar, reduction="none")
    H0 = bce_logits(logit0, logit0, reduction="none")
    H1 = bce_logits(logit1, logit1, reduction="none")

    nats = H_bar - alpha * (H0 + H1) / 2
    return nats / math.log(base)


def js_loss(
    logit0: torch.Tensor,
    logit1: torch.Tensor,
    alpha: float = 0.5,
    base: float = 2.0,
) -> torch.Tensor:
    """Consistency and confidence loss based on the Jensen-Shannon divergence."""
    return bernoulli_js(logit0, logit1, alpha=alpha, base=base).mean()


def ccs_squared_loss(logit0: torch.Tensor, logit1: torch.Tensor) -> torch.Tensor:
    """CCS loss from original paper, with squared differences between probabilities."""
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()

    consistency = torch.mean((p0 - (1 - p1)) ** 2, dim=0)
    confidence = torch.mean(torch.min(p0, p1) ** 2, dim=0)

    return consistency + confidence
