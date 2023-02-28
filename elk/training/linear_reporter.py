"""A linear ELK reporter."""

from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from simple_parsing.helpers import Serializable
from typing import Literal, Optional
import torch
import torch.nn as nn


@dataclass
class LinearReporterConfig(Serializable):
    """
    Args:
        method: The method to use for dimensionality reduction. Defaults to "pca".
        num_heads: The number of logits to output. Defaults to 1.
    """

    method: Literal["cca", "lda", "pca"] = "pca"
    num_heads: int = 1


class LinearReporter(nn.Module):
    """A linear ELK reporter, potentially with multiple heads.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    weights: torch.Tensor

    def __init__(
        self, in_features: int, cfg: LinearReporterConfig, device: Optional[str] = None
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.probe = nn.Linear(in_features, cfg.num_heads, bias=False, device=device)
        self.register_buffer("weights", torch.zeros(cfg.num_heads, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.probe(x)

    def validate_data(self, data):
        """Validate that the data's shape is valid."""
        assert len(data) == 2 and data[0].shape == data[1].shape

    def fit(self, contrast_pair: tuple[torch.Tensor, torch.Tensor]):
        """Fit the probe to the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.

        Raises:
            RuntimeError: If the best loss is not finite.
        """
        self.validate_data(contrast_pair)

        x0, x1 = contrast_pair
        diffs = torch.flatten(x0 - x1, end_dim=-2)

        if self.cfg.method == "pca":
            L, Q = torch.linalg.eigh(diffs.T.cov())
            self.probe.weight.data = Q[:, -self.cfg.num_heads :].T
            self.weights.data = L[-self.cfg.num_heads :]
        else:
            raise NotImplementedError(f"Method {self.cfg.method} not implemented.")

    @torch.no_grad()
    def score(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> list[float]:
        """Score the probe on the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.
            labels: The labels of the contrast pair.

        Returns:
            List containing the AUROC of each head on the contrast pair (x0, x1).
        """

        self.validate_data(contrast_pair)

        logit0, logit1 = map(self, contrast_pair)
        p0, p1 = logit0.sigmoid(), logit1.sigmoid()
        pred_probs = 0.5 * (p0 + (1 - p1))

        return [
            float(roc_auc_score(labels.cpu(), head_prob.cpu()))
            for head_prob in pred_probs.unbind(-1)
        ]
