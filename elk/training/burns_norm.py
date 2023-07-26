import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization  Minimal changes from the original code."""

    def forward(self, x: Tensor) -> Tensor:
        x_mean: Tensor = x - torch.mean(x, dim=0)
        std = torch.linalg.norm(x_mean, axis=0) / torch.sqrt(
            torch.tensor(x_mean.shape[0], dtype=torch.float32)
        )
        avg_norm = torch.mean(std)
        return x_mean / avg_norm
