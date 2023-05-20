"""An ELK reporter network."""

from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor, nn

from ..truncated_eigh import truncated_eigh
from .concept_eraser import ConceptEraser
from .reporter import Reporter, ReporterConfig


@dataclass
class EigenReporterConfig(ReporterConfig):
    """Configuration for an EigenReporter."""

    num_heads: int = 1
    """The number of eigenvectors to compute from the VINC matrix."""

    save_reporter_stats: bool = False
    """Whether to save the reporter statistics to disk in EigenReporter.save(). This
    is useful for debugging and analysis, but can take up a lot of disk space."""

    def __post_init__(self):
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
        contrastive_xcov_M2: Average of the unnormalized cross-covariance matrices
            across all pairs of classes (k, k').
        n: The running sum of the number of clusters processed by `update()`.
        weight: The reporter weight matrix. Guaranteed to always be orthogonal, and
            the columns are sorted in descending order of eigenvalue magnitude.
    """

    config: EigenReporterConfig
    contrastive_xcov_M2: Tensor  # negative covariance

    mean: Tensor
    n: Tensor
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
            num_classes * num_variants,
            device=device,
            dtype=dtype,
        )

        # Running statistics
        self.register_buffer(
            "mean",
            torch.zeros(in_features, device=device, dtype=dtype),
            persistent=cfg.save_reporter_stats,
        )
        self.register_buffer(
            "n",
            torch.zeros((), device=device, dtype=torch.long),
            persistent=cfg.save_reporter_stats,
        )
        self.register_buffer(
            "contrastive_xcov_M2",
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
    def consistency(self) -> Tensor:
        return -self.weight @ self.contrastive_xcov @ self.weight.mT

    @torch.no_grad()
    def update(self, hiddens: Tensor) -> None:
        (n, v, k, d) = hiddens.shape

        # Sanity checks
        assert k > 1, "Must provide at least two hidden states"
        assert hiddens.ndim == 4, "Must be of shape [batch, variants, choices, dim]"

        self.n += n * v

        # Independent indicator for each (template, pseudo-label) pair
        prompt_ids = torch.eye(k * v, device=hiddens.device).expand(n, -1, -1)
        self.norm.update(x=rearrange(hiddens, "n v k d -> n (k v) d"), y=prompt_ids)

        # Welford's online algorithm
        delta = hiddens - self.mean
        self.mean += delta.sum(dim=(0, 1, 2)) / (self.n * k)
        delta2 = hiddens - self.mean

        delta = rearrange(delta, "n v k d -> (n v) k d")
        delta2 = rearrange(delta2, "n v k d -> (n v) k d")

        # *** Negative covariance (contrastive) ***
        # Iterating over pairs of classes (k, k') where k != k'
        # TODO: Rewrite this as O(k) instead of O(k^2)
        for i, d in enumerate(delta.unbind(1)):
            for j, d_ in enumerate(delta2.unbind(1)):
                # Compare to the other classes only
                if i == j:
                    continue

                scale = 1 / (k * (k - 1))
                self.contrastive_xcov_M2.addmm_(d.mT, d_, alpha=scale)

    def fit_streaming(self, truncated: bool = False) -> float:
        """Fit the probe using the current streaming statistics."""
        # Remove the subspace responsible for pseudolabel and prompt correlations
        A = -self.norm.P @ self.contrastive_xcov @ self.norm.P.mT

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
