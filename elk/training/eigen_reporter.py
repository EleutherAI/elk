"""An ELK reporter network."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn, optim

from ..metrics import to_one_hot
from ..truncated_eigh import truncated_eigh
from ..utils.math_util import cov_mean_fused
from .reporter import Reporter, ReporterConfig


@dataclass
class EigenReporterConfig(ReporterConfig):
    """Configuration for an EigenReporter.

    Args:
        var_weight: The weight of the variance term in the loss.
        neg_cov_weight: The weight of the negative covariance term in the loss.
        num_heads: The number of reporter heads to fit. In other words, the number
            of eigenvectors to compute from the VINC matrix.
    """

    var_weight: float = 0.1
    neg_cov_weight: float = 0.5

    num_heads: int = 1

    def __post_init__(self):
        if not (0 <= self.neg_cov_weight <= 1):
            raise ValueError("neg_cov_weight must be in [0, 1]")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")


class EigenReporter(Reporter):
    """A linear reporter whose weights are computed via eigendecomposition.

    Args:
        cfg: The reporter configuration.
        in_features: The number of input features.
        num_classes: The number of classes for tracking the running means. If `None`,
            we don't track the running means at all, and the semantics of `update()`
            are a bit different. In particular, each call to `update()` is treated as a
            new dataset, with a potentially different number of classes. The covariance
            matrices are simply averaged over each batch of data passed to `update()`,
            instead of being updated with Welford's algorithm. This is useful for
            training a single reporter on multiple datasets, where the number of
            classes may vary.

    Attributes:
        config: The reporter configuration.
        intercluster_cov_M2: The unnormalized covariance matrix averaged over all
            classes.
        intracluster_cov: The running mean of the covariance matrices within each
            cluster. This doesn't need to be a running sum because it's doesn't use
            Welford's algorithm.
        contrastive_xcov_M2: Average of the unnormalized cross-covariance matrices
            across all pairs of classes (k, k').
        n: The running sum of the number of clusters processed by `update()`.
        weight: The reporter weight matrix. Guaranteed to always be orthogonal, and
            the columns are sorted in descending order of eigenvalue magnitude.
    """

    config: EigenReporterConfig

    intercluster_cov_M2: Tensor | None  # variance
    intracluster_cov: Tensor | None  # invariance
    contrastive_xcov_M2: Tensor | None  # negative covariance
    n: Tensor
    class_means: Tensor | None
    weight: Tensor

    def __init__(
        self,
        cfg: EigenReporterConfig,
        in_features: int,
        num_classes: int | None = 2,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.config = cfg

        # Learnable Platt scaling parameters
        self.bias = nn.Parameter(torch.zeros(cfg.num_heads, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(cfg.num_heads, device=device, dtype=dtype))

        # Running statistics
        self.register_buffer("n", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer(
            "class_means",
            (
                torch.zeros(num_classes, in_features, device=device, dtype=dtype)
                if num_classes is not None
                else None
            ),
        )

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

        # Reporter weights
        self.register_buffer(
            "weight",
            torch.zeros(cfg.num_heads, in_features, device=device, dtype=dtype),
        )

    def forward(self, hiddens: Tensor) -> Tensor:
        """Return the predicted log odds on input `x`."""
        raw_scores = hiddens @ self.weight.mT
        return raw_scores.mul(self.scale).add(self.bias).squeeze(-1)

    @property
    def contrastive_xcov(self) -> Tensor:
        return self.contrastive_xcov_M2 / self.n

    @property
    def intercluster_cov(self) -> Tensor:
        return self.intercluster_cov_M2 / self.n

    @property
    def confidence(self) -> Tensor:
        return self.weight @ self.intercluster_cov @ self.weight.mT

    @property
    def invariance(self) -> Tensor:
        return -self.weight @ self.intracluster_cov @ self.weight.mT

    @property
    def consistency(self) -> Tensor:
        return -self.weight @ self.contrastive_xcov @ self.weight.mT

    def clear(self) -> None:
        """Clear the running statistics of the reporter."""
        assert (
            self.contrastive_xcov_M2 is not None
            and self.intercluster_cov_M2 is not None
            and self.intracluster_cov is not None
        ), "Covariance matrices have been deleted"
        self.contrastive_xcov_M2.zero_()
        self.intracluster_cov.zero_()
        self.intercluster_cov_M2.zero_()
        self.n.zero_()

    def delete_stats(self) -> None:
        """Delete the running covariance matrices.

        This is useful for saving memory when we're done training the reporter.
        """
        self.contrastive_xcov_M2 = None
        self.intercluster_cov_M2 = None
        self.intracluster_cov = None

    @torch.no_grad()
    def update(self, hiddens: Tensor) -> None:
        assert (
            self.contrastive_xcov_M2 is not None
            and self.intercluster_cov_M2 is not None
            and self.intracluster_cov is not None
        ), "Covariance matrices have been deleted"

        (n, _, k, d) = hiddens.shape

        # Sanity checks
        assert k > 1, "Must provide at least two hidden states"
        assert hiddens.ndim == 4, "Must be of shape [batch, variants, choices, dim]"

        self.n += n

        # *** Invariance (intra-cluster) ***
        # This is just a standard online *mean* update, since we're computing the
        # mean of covariance matrices, not the covariance matrix of means.
        intra_cov = cov_mean_fused(rearrange(hiddens, "n v k d -> (n k) v d"))
        self.intracluster_cov += (n / self.n) * (intra_cov - self.intracluster_cov)

        # [n, v, k, d] -> [n, k, d]
        centroids = hiddens.mean(1)
        deltas, deltas2 = [], []

        # Iterating over classes
        for i, h in enumerate(centroids.unbind(1)):
            # Update the running means if needed
            if self.class_means is not None:
                delta = h - self.class_means[i]
                self.class_means[i] += delta.sum(dim=0) / self.n

                # Post-mean update deltas are used to update the (co)variance
                delta2 = h - self.class_means[i]  # [n, d]
            else:
                delta = h - h.mean(dim=0)
                delta2 = delta

            # *** Variance (inter-cluster) ***
            # See code at https://bit.ly/3YC9BhH and "Welford's online algorithm"
            # in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
            self.intercluster_cov_M2.addmm_(delta.mT, delta2, alpha=1 / k)
            deltas.append(delta)
            deltas2.append(delta2)

        # *** Negative covariance (contrastive) ***
        # Iterating over pairs of classes (k, k') where k != k'
        for i, d in enumerate(deltas):
            for j, d_ in enumerate(deltas2):
                # Compare to the other classes only
                if i == j:
                    continue

                scale = 1 / (k * (k - 1))
                self.contrastive_xcov_M2.addmm_(d.mT, d_, alpha=scale)

    def fit_streaming(self, truncated: bool = False) -> float:
        """Fit the probe using the current streaming statistics."""
        inv_weight = 1 - self.config.neg_cov_weight
        assert (
            self.contrastive_xcov_M2 is not None
            and self.intercluster_cov_M2 is not None
            and self.intracluster_cov is not None
        ), "Covariance matrices have been deleted"
        A = (
            self.config.var_weight * self.intercluster_cov
            - inv_weight * self.intracluster_cov
            - self.config.neg_cov_weight * self.contrastive_xcov
        )

        if truncated:
            L, Q = truncated_eigh(A, k=self.config.num_heads, seed=self.config.seed)
        else:
            try:
                L, Q = torch.linalg.eigh(A)
            except torch.linalg.LinAlgError as e:
                # Check if the matrix has non-finite values
                if not A.isfinite().all():
                    raise ValueError(
                        "Fitting the reporter failed because the VINC matrix has "
                        "non-finite entries. Usually this means the hidden states "
                        "themselves had non-finite values."
                    ) from e
                else:
                    raise e

            L, Q = L[-self.config.num_heads :], Q[:, -self.config.num_heads :]

        self.weight.data = Q.T
        return -float(L[-1])

    def fit(
        self,
        hiddens: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        """Fit the probe to the contrast set `hiddens`.

        Args:
            hiddens: The contrast set of shape [batch, variants, choices, dim].
            labels: The ground truth labels if available.

        Returns:
            loss: Negative eigenvalue associated with the VINC direction.
        """
        self.update(hiddens)
        loss = self.fit_streaming()

        if labels is not None:
            (_, v, k, _) = hiddens.shape
            hiddens = rearrange(hiddens, "n v k d -> (n v k) d")
            labels = to_one_hot(repeat(labels, "n -> (n v)", v=v), k).flatten()

            self.platt_scale(labels, hiddens)

        return loss

    def platt_scale(self, labels: Tensor, hiddens: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            labels: Binary labels of shape [batch].
            hiddens: Hidden states of shape [batch, dim].
            max_iter: Maximum number of iterations for LBFGS.
        """
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(hiddens.dtype).eps,
            tolerance_grad=torch.finfo(hiddens.dtype).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self(hiddens), labels.float()
            )

            loss.backward()
            return float(loss)

        opt.step(closure)

    def save(self, path: Path | str, save_stats=False):
        # TODO: this method will save separate JSON and PT files
        if not save_stats:
            self.delete_stats()
        super().save(path)
