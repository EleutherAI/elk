"""An ELK reporter network."""

from ..math_util import cov_mean_fused
from .reporter import Reporter, ReporterConfig
from dataclasses import dataclass
from torch import Tensor
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class EigenReporterConfig(ReporterConfig):
    inv_weight: float = 5.0
    neg_cov_weight: float = 5.0


class EigenReporter(Reporter):
    """A linear reporter whose weights are computed via eigendecomposition.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    config: EigenReporterConfig

    n: Tensor

    contrastive_M2: Tensor
    M2: Tensor
    invariance: Tensor

    def __init__(
        self,
        in_features: int,
        cfg: EigenReporterConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, cfg, device=device, dtype=dtype)

        self.linear = nn.Linear(in_features, 1, bias=False, device=device, dtype=dtype)

        self.register_buffer(
            "contrastive_M2",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "M2", torch.zeros(in_features, in_features, device=device, dtype=dtype)
        )
        self.register_buffer(
            "invariance",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.linear(x).squeeze(-1)

    def predict(self, x_pos: Tensor, x_neg: Tensor) -> Tensor:
        return 0.5 * (self(x_pos) - self(x_neg))

    @property
    def neg_covariance(self) -> Tensor:
        return self.contrastive_M2 / self.n

    @property
    def variance(self) -> Tensor:
        return self.M2 / self.n

    @torch.no_grad()
    def update(self, x_pos: Tensor, x_neg: Tensor) -> None:
        # Sanity checks
        assert x_pos.ndim == 3, "x_pos must be of shape [batch, num_variants, d]"
        assert x_pos.shape == x_neg.shape, "x_pos and x_neg must have the same shape"

        # Average across variants inside each cluster, computing the centroids.
        pos_centroids, neg_centroids = x_pos.mean(1), x_neg.mean(1)

        # We don't actually call super because we need access to the earlier estimate
        # of the population mean in order to update (cross-)covariances properly
        # super().update(x_pos, x_neg)

        sample_n = pos_centroids.shape[0]
        self.n += sample_n

        # Update the running means; super().update() does this usually
        neg_delta = neg_centroids - self.neg_mean
        pos_delta = pos_centroids - self.pos_mean
        self.neg_mean += neg_delta.sum(dim=0) / self.n
        self.pos_mean += pos_delta.sum(dim=0) / self.n

        # *** Variance (inter-cluster) ***
        # See code at https://bit.ly/3YC9BhH, as well as "Welford's online algorithm"
        # in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
        # Post-mean update deltas are used to update the (co)variance
        neg_delta2 = neg_centroids - self.neg_mean  # [n, d]
        pos_delta2 = pos_centroids - self.pos_mean  # [n, d]
        self.M2.addmm_(neg_delta.mT, neg_delta2)
        self.M2.addmm_(pos_delta.mT, pos_delta2)

        # *** Invariance (intra-cluster) ***
        # This is just a standard online *mean* update, since we're computing the
        # mean of covariance matrices, not the covariance matrix of means.
        sample_invar = cov_mean_fused(x_pos) + cov_mean_fused(x_neg)
        self.invariance += (sample_n / self.n) * (sample_invar - self.invariance)

        # *** Negative covariance ***
        self.contrastive_M2.addmm_(neg_delta.mT, pos_delta2)
        self.contrastive_M2.addmm_(pos_delta.mT, neg_delta2)

    def fit_streaming(self) -> float:
        """Fit the probe using the current streaming statistics."""
        alpha, beta = self.config.inv_weight, self.config.neg_cov_weight
        A = self.variance - alpha * self.invariance - beta * self.neg_covariance

        # Use SciPy's sparse eigensolver for CPU tensors. This is a frontend to ARPACK,
        # which uses the Lanczos method under the hood.
        if A.device.type == "cpu":
            from scipy.sparse.linalg import eigsh

            L, Q = eigsh(A.numpy(), k=1)
            self.linear.weight.data = torch.from_numpy(Q).T
        else:
            L, Q = torch.linalg.eigh(A)
            self.linear.weight.data = Q[:, -1, None].T

        return float(L[-1])

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
        self.update(x_pos, x_neg)
        return self.fit_streaming()
