import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization. Minimal changes from the original code."""

    def __init__(self, scale: bool = True):
        super().__init__()
        self.scale: bool = scale

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes per prompt template
        Args:
            x: input of dimension (n, v, c, d) or (n, v, d)
        Returns:
            x_normalized: normalized output
        """
        num_elements = x.shape[0]
        x_normalized: Tensor = x - x.mean(dim=0) if num_elements > 1 else x

        if not self.scale:
            return x_normalized
        else:
            std = torch.linalg.norm(x_normalized, dim=0) / x_normalized.shape[0] ** 0.5
            assert std.dim() == x.dim() - 1

            # Compute the dimensions over which
            # we want to compute the mean standard deviation
            # exclude the first dimension v,
            # which is the template dimension
            dims = tuple(range(1, std.dim()))

            avg_norm = std.mean(dim=dims, keepdim=True)

            return x_normalized / avg_norm
