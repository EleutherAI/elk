"""An ELK reporter network."""

from ..math_util import cov_mean_fused
from .reporter import EvalResult, Reporter, ReporterConfig
from copy import deepcopy
from dataclasses import dataclass
from torch import nn, optim, Tensor
from typing import Optional, Sequence
import torch


@dataclass
class EigenReporterConfig(ReporterConfig):
    """Configuration for an EigenReporter.

    Args:
        var_weight: The weight of the variance term in the loss.
        inv_weight: The weight of the invariance term in the loss.
        neg_cov_weight: The weight of the negative covariance term in the loss.
        num_heads: The number of reporter heads to fit. In other words, the number
            of eigenvectors to compute from the VINC matrix.
    """

    var_weight: float = 1.0
    inv_weight: float = 5.0
    neg_cov_weight: float = 5.0

    num_heads: int = 1


class EigenReporter(Reporter):
    """A linear reporter whose weights are computed via eigendecomposition.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.

    Attributes:
        config: The reporter configuration.
        intercluster_cov_M2: The running sum of the covariance matrices of the
            centroids of the positive and negative clusters.
        intracluster_cov: The running mean of the covariance matrices within each
            cluster. This doesn't need to be a running sum because it's doesn't use
            Welford's algorithm.
        contrastive_xcov_M2: The running sum of the cross-covariance between the
            centroids of the positive and negative clusters.
        n: The running sum of the number of samples in the positive and negative
            clusters.
        weight: The reporter weight matrix. Guaranteed to always be orthogonal, and
            the columns are sorted in descending order of eigenvalue magnitude.
    """

    config: EigenReporterConfig

    intercluster_cov_M2: Tensor  # variance
    intracluster_cov: Tensor  # invariance
    contrastive_xcov_M2: Tensor  # negative covariance
    n: Tensor
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        cfg: EigenReporterConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(in_features, cfg, device=device, dtype=dtype)

        # Learnable Platt scaling parameters
        self.bias = nn.Parameter(torch.zeros(cfg.num_heads, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(cfg.num_heads, device=device, dtype=dtype))

        self.register_buffer(
            "contrastive_xcov_M2",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "intercluster_cov_M2",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "intracluster_cov",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "weight",
            torch.zeros(cfg.num_heads, in_features, device=device, dtype=dtype),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the predicted log odds on input `x`."""
        raw_scores = x @ self.weight.mT
        return raw_scores.mul(self.scale).add(self.bias).squeeze(-1)

    def predict(self, x_pos: Tensor, x_neg: Tensor) -> Tensor:
        """Return the predicted log odds on the contrast pair `(x_pos, x_neg)`."""
        return 0.5 * (self(x_pos) - self(x_neg))

    @property
    def contrastive_xcov(self) -> Tensor:
        return self.contrastive_xcov_M2 / self.n

    @property
    def intercluster_cov(self) -> Tensor:
        return self.intercluster_cov_M2 / self.n

    def clear(self) -> None:
        """Clear the running statistics of the reporter."""
        self.contrastive_xcov_M2.zero_()
        self.intracluster_cov.zero_()
        self.intercluster_cov_M2.zero_()
        self.n.zero_()

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
        self.intercluster_cov_M2.addmm_(neg_delta.mT, neg_delta2)
        self.intercluster_cov_M2.addmm_(pos_delta.mT, pos_delta2)

        # *** Invariance (intra-cluster) ***
        # This is just a standard online *mean* update, since we're computing the
        # mean of covariance matrices, not the covariance matrix of means.
        sample_invar = cov_mean_fused(x_pos) + cov_mean_fused(x_neg)
        self.intracluster_cov += (sample_n / self.n) * (
            sample_invar - self.intracluster_cov
        )

        # *** Negative covariance ***
        self.contrastive_xcov_M2.addmm_(neg_delta.mT, pos_delta2)
        self.contrastive_xcov_M2.addmm_(pos_delta.mT, neg_delta2)

    def fit_streaming(self, warm_start: bool = False) -> float:
        """Fit the probe using the current streaming statistics."""
        A = (
            self.config.var_weight * self.intercluster_cov
            - self.config.inv_weight * self.intracluster_cov
            - self.config.neg_cov_weight * self.contrastive_xcov
        )

        # Use SciPy's sparse eigensolver for CPU tensors. This is a frontend to ARPACK,
        # which uses the Lanczos method under the hood.
        if A.device.type == "cpu":
            from scipy.sparse.linalg import eigsh

            v0 = self.weight.T.numpy() if warm_start else None

            # We use "LA" (largest algebraic) instead of "LM" (largest magnitude) to
            # ensure that the eigenvalue is positive and not a large negative one
            L, Q = eigsh(A.numpy(), k=self.config.num_heads, v0=v0, which="LA")
            self.weight.data = torch.from_numpy(Q).T
        else:
            L, Q = torch.linalg.eigh(A)
            self.weight.data = Q[:, -self.config.num_heads :].T

        return -float(L[-1])

    def fit(
        self,
        x_pos: Tensor,
        x_neg: Tensor,
        labels: Optional[Tensor] = None,
        *,
        platt_scale: bool = True,
    ) -> float:
        """Fit the probe to the contrast pair (x_pos, x_neg).

        Args:
            x_pos: The positive examples.
            x_neg: The negative examples.
            labels: The ground truth labels if available.
            platt_scale: Whether to fit the scale and bias terms to data with LBFGS.
                This is only used if labels are available.

        Returns:
            loss: Negative eigenvalue associated with the VINC direction.
        """
        assert x_pos.shape == x_neg.shape
        self.update(x_pos, x_neg)
        loss = self.fit_streaming()
        if labels is not None and platt_scale:
            self.platt_scale(labels, x_pos, x_neg)

        return loss

    def platt_scale(
        self, labels: Tensor, x_pos: Tensor, x_neg: Tensor, max_iter: int = 100
    ):
        """Fit the scale and bias terms to data with LBFGS."""

        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(x_pos.dtype).eps,
            tolerance_grad=torch.finfo(x_pos.dtype).eps,
        )
        labels = labels.repeat_interleave(x_pos.shape[1]).float()

        def closure():
            opt.zero_grad()
            logits = self.predict(x_pos, x_neg).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            return float(loss)

        opt.step(closure)
