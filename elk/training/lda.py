"""An ELK reporter network."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from concept_erasure import optimal_linear_shrinkage
from einops import rearrange
from torch import Tensor

from ..utils.math_util import cov, cov_mean_fused
from .common import FitterConfig, Reporter


@dataclass
class LdaConfig(FitterConfig):
    """Configuration for an LdaFitter."""

    anchor_gamma: float = 1.0
    """Gamma parameter for anchor regression."""

    invariance_weight: float = 0.5
    """Weight of the prompt invariance term in the loss."""

    l2_penalty: float = 0.0

    def __post_init__(self):
        assert self.anchor_gamma >= 0, "anchor_gamma must be non-negative"
        assert 0 <= self.invariance_weight <= 1, "invariance_weight must be in [0, 1]"
        assert self.l2_penalty >= 0, "l2_penalty must be non-negative"


class LdaFitter:
    """Linear Discriminant Analysis (LDA)"""

    config: LdaConfig

    def __init__(self, cfg: LdaConfig):
        super().__init__()
        self.config = cfg

    def fit(self, hiddens: Tensor, labels: Tensor) -> Reporter:
        """Fit the probe to the contrast set `hiddens`.

        Args:
            hiddens: The contrast set of shape [batch, variants, (choices,) dim].
            labels: Integer labels of shape [batch].
        """
        n, v, *_ = hiddens.shape
        assert n == labels.shape[0], "hiddens and labels must have the same batch size"

        # This is a contrast set; create a true-false label for each element
        if len(hiddens.shape) == 4:
            hiddens = rearrange(hiddens, "n v k d -> (n k) v d")
            labels = F.one_hot(labels.long()).flatten()

            n = len(labels)
            counts = (labels.sum(), n - labels.sum())
        else:
            counts = torch.bincount(labels)
            assert len(counts) == 2, "Only binary classification is supported for now"

        # Construct targets for the least-squares dual problem
        z = torch.where(labels.bool(), n / counts[0], -n / counts[1]).unsqueeze(1)

        # Adjust X and Z for anchor regression <https://arxiv.org/abs/1801.06229>
        gamma = self.config.anchor_gamma
        if gamma != 1.0:
            # Implicitly compute n x n orthogonal projection onto the column space of
            # the anchor variables without materializing the whole matrix. Since the
            # anchors are one-hot, it turns out this is equivalent to adding a multiple
            # of the anchor-conditional means.
            # In general you're supposed to adjust the labels too, but we don't need
            # to do that because by construction the anchor-conditional means of the
            # labels are already all zero.
            hiddens = hiddens + (gamma**0.5 - 1) * hiddens.mean(0)

        # We can decompose the covariance matrix into the sum of the within-cluster
        # covariance and the between-cluster covariance. This allows us to put extra
        # weight on the within-cluster variance to encourage invariance to the prompt.
        # NOTE: We're not applying shrinkage to each cluster covariance matrix because
        # we're averaging over them, which should reduce the variance of the estimate
        # a lot. Shrinkage could make MSE worse in this case.
        S_between = optimal_linear_shrinkage(cov(hiddens.mean(1)), n)
        S_within = cov_mean_fused(hiddens)

        # Convex combination but multiply by 2 to keep the same scale
        alpha = 2 * self.config.invariance_weight
        S = alpha * S_within + (2 - alpha) * S_between

        # Add ridge penalty
        torch.linalg.diagonal(S).add_(self.config.l2_penalty)

        # Broadcast the labels across variants
        sigma_xz = cov(hiddens, z.expand_as(hiddens[..., 0]).unsqueeze(-1))
        w = torch.linalg.solve(S, sigma_xz.squeeze(-1))

        return Reporter(w[None])
