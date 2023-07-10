"""An ELK reporter network."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from concept_erasure import optimal_linear_shrinkage
from einops import rearrange
from torch import Tensor

from .common import FitterConfig, Reporter


@dataclass
class LdaConfig(FitterConfig):
    """Configuration for an LdaFitter."""

    l2_penalty: float = 0.0


class LdaFitter:
    """Linear Discriminant Analysis (LDA)"""

    config: LdaConfig

    def __init__(self, cfg: LdaConfig):
        super().__init__()
        self.config = cfg

    def fit(self, hiddens: Tensor, labels: Tensor) -> Reporter:
        """Fit the probe to the contrast set `hiddens`.

        Args:
            hiddens: The contrast set of shape [batch, variants, choices, dim].
        """
        (n, _, k, _) = hiddens.shape

        # Sanity checks
        assert k > 1, "Must provide at least two hidden states"
        assert hiddens.ndim == 4, "Must be of shape [batch, variants, choices, dim]"

        # Create a true-false label for each element of the contrast set
        mask = F.one_hot(labels.long(), num_classes=k).bool()  # [n, k]
        mask = mask[:, None, :].expand_as(hiddens[..., 0])  # [n, 1, k] -> [n, v, k]

        mu_pos = hiddens[mask].mean(0)
        mu_neg = hiddens[~mask].mean(0)

        sigma = rearrange(hiddens, "n v k d -> d (n v k)").cov()
        sigma = optimal_linear_shrinkage(sigma, n)
        torch.linalg.diagonal(sigma).add_(self.config.l2_penalty)

        w = torch.linalg.solve(sigma, mu_pos - mu_neg)
        return Reporter(w[None])
