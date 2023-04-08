"""An ELK reporter network."""

from dataclasses import dataclass
from typing import Optional
from warnings import warn

import torch
from einops import rearrange, repeat
from torch import Tensor, nn, optim

from ..math_util import cov_mean_fused
from ..truncated_eigh import ConvergenceError, truncated_eigh
from .reporter import Reporter, ReporterConfig


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
        cfg: EigenReporterConfig,
        in_features: int,
        num_classes: int = 2,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(cfg, in_features, num_classes, device=device, dtype=dtype)

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

    def predict(self, *hiddens: Tensor) -> Tensor:
        """Return the predicted logits on the contrast set `hiddens`."""
        # breakpoint()
        if len(hiddens) == 1:
            return self(hiddens[0])

        elif len(hiddens) == 2:
            return 0.5 * (self(hiddens[0]) - self(hiddens[1]))
        else:
            return torch.stack(list(map(self, hiddens)), dim=-1)

    def predict_prob(self, *hiddens: Tensor) -> Tensor:
        """Return the predicted probabilities on the contrast set `hiddens`."""
        logits = self.predict(*hiddens)
        if len(hiddens) == 2:
            return logits.sigmoid()
        else:
            return logits.softmax(dim=-1)

    @property
    def contrastive_xcov(self) -> Tensor:
        return self.contrastive_xcov_M2 / self.n

    @property
    def intercluster_cov(self) -> Tensor:
        return self.intercluster_cov_M2 / self.n

    @property
    def confidence(self) -> Tensor:
        return self.weight.mT @ self.intercluster_cov @ self.weight

    @property
    def invariance(self) -> Tensor:
        return -self.weight.mT @ self.intracluster_cov @ self.weight

    @property
    def consistency(self) -> Tensor:
        return -self.weight.mT @ self.contrastive_xcov @ self.weight

    def clear(self) -> None:
        """Clear the running statistics of the reporter."""
        self.contrastive_xcov_M2.zero_()
        self.intracluster_cov.zero_()
        self.intercluster_cov_M2.zero_()
        self.n.zero_()

    @torch.no_grad()
    def update(self, *hiddens: Tensor) -> None:
        k = len(hiddens)
        assert k > 1, "Must provide at least two hidden states"

        # Sanity checks
        pivot, *rest = hiddens
        assert pivot.ndim == 3, "hidden must be of shape [batch, num_variants, d]"
        for h in rest:
            assert h.shape == pivot.shape, "All hiddens must have the same shape"

        # We don't actually call super because we need access to the earlier estimate
        # of the population mean in order to update (cross-)covariances properly
        # super().update(x_pos, x_neg)

        sample_n = pivot.shape[0]
        self.n += sample_n

        # *** Invariance (intra-cluster) ***
        # This is just a standard online *mean* update, since we're computing the
        # mean of covariance matrices, not the covariance matrix of means.
        sample_invar = sum(map(cov_mean_fused, hiddens)) / k
        self.intracluster_cov += (sample_n / self.n) * (
            sample_invar - self.intracluster_cov
        )

        # [n, v, d] -> [n, d]
        centroids = [h.mean(1) for h in hiddens]
        deltas, deltas2 = [], []

        for i, h in enumerate(centroids):
            # Update the running means; super().update() does this usually
            delta = h - self.class_means[i]
            self.class_means[i] += delta.sum(dim=0) / self.n

            # *** Variance (inter-cluster) ***
            # See code at https://bit.ly/3YC9BhH and "Welford's online algorithm"
            # in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
            # Post-mean update deltas are used to update the (co)variance
            delta2 = h - self.class_means[i]  # [n, d]
            self.intercluster_cov_M2.addmm_(delta.mT, delta2, alpha=1 / k)
            deltas.append(delta)
            deltas2.append(delta2)

        # *** Negative covariance (contrastive) ***
        for i, d in enumerate(deltas):
            for j, d_ in enumerate(deltas2):
                # Compare to the other classes only
                if i == j:
                    continue

                scale = 1 / (k * (k - 1))
                self.contrastive_xcov_M2.addmm_(d.mT, d_, alpha=scale)

    def fit_streaming(self) -> float:
        """Fit the probe using the current streaming statistics."""
        A = (
            self.config.var_weight * self.intercluster_cov
            - self.config.inv_weight * self.intracluster_cov
            - self.config.neg_cov_weight * self.contrastive_xcov
        )

        try:
            L, Q = truncated_eigh(A, k=self.config.num_heads)
        except (ConvergenceError, RuntimeError):
            warn(
                "Truncated eigendecomposition failed to converge. Falling back on "
                "PyTorch's dense eigensolver."
            )

            L, Q = torch.linalg.eigh(A)
            L, Q = L[-self.config.num_heads :], Q[:, -self.config.num_heads :]

        self.weight.data = Q.T
        return -float(L[-1])

    def fit(
        self,
        *hiddens: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        """Fit the probe to the contrast set `hiddens`.

        Args:
            hiddens: The contrast set of hidden states.
            labels: The ground truth labels if available.

        Returns:
            loss: Negative eigenvalue associated with the VINC direction.
        """
        self.update(*hiddens)
        loss = self.fit_streaming()

        if labels is not None:
            self.platt_scale(labels, *hiddens)

        return loss

    def platt_scale(self, labels: Tensor, *hiddens: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS."""

        pivot, *_ = hiddens
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(pivot.dtype).eps,
            tolerance_grad=torch.finfo(pivot.dtype).eps,
        )
        labels = repeat(labels, "n -> (n v)", v=pivot.shape[1])

        def closure():
            opt.zero_grad()
            logits = rearrange(self.predict(*hiddens), "n v ... -> (n v) ...")
            if len(logits.shape) == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float()
                )
            else:
                loss = nn.functional.cross_entropy(logits, labels.long())

            loss.backward()
            return float(loss)

        opt.step(closure)
