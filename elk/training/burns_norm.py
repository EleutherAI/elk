import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization  Minimal changes from the original code."""

    def forward(self, x_all: Tensor) -> Tensor:
        assert x_all.ndim == 4
        x_mean = x_all - x_all.mean(dim=0, keepdim=True)
        std = torch.linalg.norm(x_mean, dim=0, keepdim=True) / torch.sqrt(
            torch.tensor(x_mean.shape[0], dtype=torch.float32)
        )
        avg_norm = std.mean(dim=(0, 2, 3), keepdim=True)
        return x_mean / avg_norm
