"""An ELK reporter network."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .reporter import Reporter, ReporterConfig


@dataclass
class LdaReporterConfig(ReporterConfig):
    """Configuration for an LdaReporter."""

    @classmethod
    def reporter_class(cls) -> type[Reporter]:
        return LdaReporter


class LdaReporter(Reporter):
    """Linear Discriminant Analysis (LDA) reporter.

    Args:
        cfg: The reporter configuration.
        in_features: The number of input features.
        num_classes: The number of classes for tracking the running means.

    Attributes:
        config: The reporter configuration.
        n: The running sum of the number of clusters processed by `update()`.
        weight: The reporter weight matrix. Guaranteed to always be orthogonal, and
            the columns are sorted in descending order of eigenvalue magnitude.
    """

    config: LdaReporterConfig

    n: Tensor
    class_means: Tensor
    weight: Tensor

    def __init__(
        self,
        cfg: LdaReporterConfig,
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
        self.bias = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        # Running statistics
        self.register_buffer(
            "class_means",
            torch.zeros(num_classes, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "sample_sizes",
            torch.zeros((), device=device, dtype=torch.long),
        )
        # Reporter weights
        self.register_buffer(
            "weight",
            torch.zeros(1, in_features, device=device, dtype=dtype),
        )

    def forward(self, hiddens: Tensor) -> Tensor:
        """Return the predicted log odds on input `x`."""
        raw_scores = hiddens @ self.weight.mT
        return raw_scores.mul(self.scale).add(self.bias).squeeze(-1)

    def fit(self, hiddens: Tensor, labels: Tensor):
        """Fit the probe to the contrast set `hiddens`.

        Args:
            hiddens: The contrast set of shape [batch, variants, choices, dim].

        Returns:
            loss: Negative eigenvalue associated with the VINC direction.
        """
        (n, v, k, d) = hiddens.shape

        # Sanity checks
        assert k > 1, "Must provide at least two hidden states"
        assert hiddens.ndim == 4, "Must be of shape [batch, variants, choices, dim]"

        # Create a true-false label for each element of the contrast set
        mask = F.one_hot(labels.long(), num_classes=k).bool()  # [n, k]
        mask = mask[:, None, :].expand_as(hiddens[..., 0])  # [n, 1, k] -> [n, v, k]

        mu_pos = hiddens[mask].mean(0)
        mu_neg = hiddens[~mask].mean(0)
        sigma = rearrange(hiddens, "n v k d -> d (n v k)").cov()

        w = torch.linalg.pinv(sigma) @ (mu_pos - mu_neg)
        self.weight.data = w[None]

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
