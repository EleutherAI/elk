"""An ELK reporter network."""

from dataclasses import dataclass
from pathlib import Path

import torch
from concept_erasure import ConceptEraser
from einops import rearrange
from torch import Tensor, nn

from ..truncated_eigh import truncated_eigh
from ..utils.math_util import cov_mean_fused
from .reporter import Reporter, ReporterConfig


@dataclass
class EigenReporterConfig(ReporterConfig):
    """Configuration for an EigenReporter."""

    var_weight: float = 0.0
    """The weight of the variance term in the loss."""

    neg_cov_weight: float = 0.5
    """The weight of the negative covariance term in the loss."""

    num_heads: int = 1
    """The number of eigenvectors to compute from the VINC matrix."""

    save_reporter_stats: bool = False
    """Whether to save the reporter statistics to disk in EigenReporter.save(). This
    is useful for debugging and analysis, but can take up a lot of disk space."""

    erase_prompts: bool = False
    """Whether to apply concept erasure on the prompt template IDs."""

    use_centroids: bool = True
    """Whether to average hiddens within each cluster before computing covariance."""

    def __post_init__(self):
        if not (0 <= self.neg_cov_weight <= 1):
            raise ValueError("neg_cov_weight must be in [0, 1]")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

    @classmethod
    def reporter_class(cls) -> type[Reporter]:
        return EigenReporter


class EigenReporter(Reporter):
    """A linear reporter whose weights are computed via eigendecomposition.

    Args:
        cfg: The reporter configuration.
        in_features: The number of input features.
        num_classes: The number of classes for tracking the running means.

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

    intercluster_cov_M2: Tensor  # variance
    intracluster_cov: Tensor  # invariance
    contrastive_xcov_M2: Tensor  # negative covariance

    n: Tensor
    class_means: Tensor | None
    weight: Tensor

    def __init__(
        self,
        cfg: EigenReporterConfig,
        in_features: int,
        num_classes: int = 2,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_variants: int = 1,
    ):
        super().__init__()
        self.config = cfg
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_variants = num_variants

        # Learnable Platt scaling parameters
        self.bias = nn.Parameter(torch.zeros(cfg.num_heads, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(cfg.num_heads, device=device, dtype=dtype))
        self.norm = ConceptEraser(
            in_features,
            num_classes * num_variants if cfg.erase_prompts else num_classes,
            device=device,
            dtype=dtype,
        )

        # Running statistics
        self.register_buffer(
            "n",
            torch.zeros((), device=device, dtype=torch.long),
            persistent=cfg.save_reporter_stats,
        )
        self.register_buffer(
            "class_means",
            (
                torch.zeros(num_classes, in_features, device=device, dtype=dtype)
                if num_classes is not None
                else None
            ),
            persistent=cfg.save_reporter_stats,
        )

        self.register_buffer(
            "contrastive_xcov_M2",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
            persistent=cfg.save_reporter_stats,
        )
        self.register_buffer(
            "intercluster_cov_M2",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
            persistent=cfg.save_reporter_stats,
        )
        self.register_buffer(
            "intracluster_cov",
            torch.zeros(in_features, in_features, device=device, dtype=dtype),
            persistent=cfg.save_reporter_stats,
        )

        # Reporter weights
        self.register_buffer(
            "weight",
            torch.zeros(cfg.num_heads, in_features, device=device, dtype=dtype),
        )

    def forward(self, hiddens: Tensor) -> Tensor:
        """Return the predicted log odds on input `x`."""
        raw_scores = self.norm(hiddens) @ self.weight.mT
        return raw_scores.mul(self.scale).add(self.bias).squeeze(-1)

    @property
    def contrastive_xcov(self) -> Tensor:
        assert self.n > 0, "Stats not initialized; did you set save_reporter_stats?"
        return self.contrastive_xcov_M2 / self.n

    @property
    def intercluster_cov(self) -> Tensor:
        assert self.n > 0, "Stats not initialized; did you set save_reporter_stats?"
        return self.intercluster_cov_M2 / self.n

    @property
    def confidence(self) -> Tensor:
        return self.weight @ self.intercluster_cov @ self.weight.mT

    @property
    def invariance(self) -> Tensor:
        assert self.n > 0, "Stats not initialized; did you set save_reporter_stats?"
        return -self.weight @ self.intracluster_cov @ self.weight.mT

    @property
    def consistency(self) -> Tensor:
        return -self.weight @ self.contrastive_xcov @ self.weight.mT

    @torch.no_grad()
    def update(self, hiddens: Tensor) -> None:
        (n, v, k, d) = hiddens.shape

        # Sanity checks
        assert k > 1, "Must provide at least two hidden states"
        assert hiddens.ndim == 4, "Must be of shape [batch, variants, choices, dim]"

        self.n += n

        if self.config.erase_prompts:
            # Independent indicator for each (template, pseudo-label) pair
            indicators = torch.eye(k * v, device=hiddens.device).expand(n, -1, -1)
            self.norm.update(x=hiddens, z=indicators)
        else:
            # Only use indicators for each pseudo-label
            indicators = torch.eye(k, device=hiddens.device).expand(n, v, -1, -1)

        self.norm.update(x=hiddens, z=indicators)

        # *** Invariance (intra-cluster) ***
        # This is just a standard online *mean* update, since we're computing the
        # mean of covariance matrices, not the covariance matrix of means.
        intra_cov = cov_mean_fused(rearrange(hiddens, "n v k d -> (n k) v d"))
        self.intracluster_cov += (n / self.n) * (intra_cov - self.intracluster_cov)

        if self.config.use_centroids:
            # VINC style
            centroids = hiddens.mean(1)
        else:
            # CRC-TPC style
            centroids = rearrange(hiddens, "n v k d -> (n v) k d")

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
        A = (
            self.config.var_weight * self.intercluster_cov
            - inv_weight * self.intracluster_cov
            - self.config.neg_cov_weight * self.contrastive_xcov
        )

        # Remove the subspace responsible for pseudolabel correlations
        A = self.norm.P @ A @ self.norm.P.mT

        if truncated:
            L, Q = truncated_eigh(A, k=self.config.num_heads, seed=self.config.seed)
        else:
            try:
                L, Q = torch.linalg.eigh(A)
            except torch.linalg.LinAlgError:
                try:
                    L, Q = torch.linalg.eig(A)
                    L, Q = L.real, Q.real
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

    def fit(self, hiddens: Tensor) -> float:
        """Fit the probe to the contrast set `hiddens`.

        Args:
            hiddens: The contrast set of shape [batch, variants, choices, dim].

        Returns:
            loss: Negative eigenvalue associated with the VINC direction.
        """
        self.update(hiddens)
        return self.fit_streaming()

    def save(self, path: Path | str) -> None:
        """Save the reporter to a file."""
        # We basically never want to instantiate the reporter on the same device
        # it happened to be trained on, so we save the state dict as CPU tensors.
        # Bizarrely, this also seems to save a LOT of disk space in some cases.
        state = {k: v.cpu() for k, v in self.state_dict().items()}
        state.update(
            in_features=self.in_features,
            num_classes=self.num_classes,
            num_variants=self.num_variants,
        )
        torch.save(state, path)
