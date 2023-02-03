import torch


def ccs_squared_loss(logit0: torch.Tensor, logit1: torch.Tensor) -> torch.Tensor:
    """CCS loss from original paper, with squared differences between probabilities."""
    p0, p1 = logit0.sigmoid(), logit1.sigmoid()

    consistency = torch.mean((p0 - (1 - p1)) ** 2, dim=0)
    confidence = torch.mean(torch.min(p0, p1) ** 2, dim=0)

    return consistency + confidence
