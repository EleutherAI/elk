"""A linear ELK reporter."""

from .reporter import OptimConfig, Reporter, ReporterConfig
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from typing import Literal, Optional
import torch
import torch.nn as nn


@dataclass
class LinearReporterConfig(ReporterConfig):
    init: Literal["pca", "random", "zero"] = "pca"


class LinearReporter(Reporter):
    """A linear ELK reporter, potentially with multiple heads.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    config_cls = LinearReporterConfig
    config: LinearReporterConfig
    head_weights: torch.Tensor

    def __init__(
        self, in_features: int, cfg: LinearReporterConfig, device: Optional[str] = None
    ):
        super().__init__(in_features, cfg, device)

        self.config = cfg
        self.device = device
        self.linear = nn.Linear(in_features, cfg.num_heads, bias=False, device=device)
        self.register_buffer("head_weights", torch.zeros(cfg.num_heads, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.linear(x)

    def fit(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        cfg: OptimConfig = OptimConfig(),
    ):
        self.validate_data(contrast_pair)

        x0, x1 = contrast_pair

        if self.config.init == "pca":
            diffs = torch.flatten(x0 - x1, end_dim=-2)
            _, S, V = torch.pca_lowrank(diffs, q=self.config.num_heads)

            self.head_weights.data = S
            self.linear.weight.data = V.T

        elif self.config.init == "random":
            self.linear.reset_parameters()
        elif self.config.init == "zero":
            self.linear.bias.data.zero_()
            self.linear.weight.data.zero_()
