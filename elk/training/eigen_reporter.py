"""An ELK reporter network."""

from ..math_util import batch_cov
from .reporter import Reporter, ReporterConfig
from dataclasses import dataclass
from torch import Tensor
from typing import Literal, Optional
import torch
import torch.nn as nn


@dataclass
class EigenReporterConfig(ReporterConfig):
    """ """

    inv_weight: float = 1.0
    neg_cov_weight: float = 1.0
    solver: Literal["arpack", "dense", "power"] = "dense"


class EigenReporter(Reporter):
    """A linear reporter whose weights are computed via eigendecomposition.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    config: EigenReporterConfig

    def __init__(
        self, in_features: int, cfg: EigenReporterConfig, device: Optional[str] = None
    ):
        super().__init__()

        self.config = cfg
        self.linear = nn.Linear(in_features, 1, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.linear(x).squeeze(-1)

    def predict(self, x_pos: Tensor, x_neg: Tensor) -> Tensor:
        return 0.5 * (self(x_pos) - self(x_neg))

    def fit(
        self,
        x_pos: Tensor,
        x_neg: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        """Fit the probe to the contrast pair (x_pos, x_neg).

        Args:
            x_pos: The positive examples.
            x_neg: The negative examples.

        Returns:
            loss: The best loss obtained.
        """
        assert x_pos.shape == x_neg.shape

        # Variance
        pos_bar, neg_bar = x_pos.mean(1), x_neg.mean(1)  # [batch, d]
        inter_variance = batch_cov(pos_bar) + batch_cov(neg_bar)  # [d, d]

        # Invariance
        intra_variance = batch_cov(x_pos).mean(0) + batch_cov(x_neg).mean(0)  # [d, d]

        # Negative covariance
        contrastive_variance = pos_bar.mT @ neg_bar + neg_bar.mT @ pos_bar  # [d, d]

        alpha, beta = self.config.inv_weight, self.config.neg_cov_weight
        A = inter_variance - alpha * intra_variance - beta * contrastive_variance

        L, Q = torch.linalg.eigh(A)
        self.linear.weight.data = Q[:, -1, None]

        return L[-1]
