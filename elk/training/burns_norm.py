import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization  Minimal changes from the original code."""

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() == 3
        ), f"the input should have a dimension of 3 not dimension {x.dim()}, \
        current shape of input x: {x.shape}"

        x_mean: Tensor = x - torch.mean(x, dim=0)
        norm = torch.linalg.norm(x_mean, dim=2)
        avg_norm = torch.mean(norm)
        eps = torch.finfo(x.dtype).eps
        return (
            x_mean
            / (avg_norm + eps)  # to avoid division by zero
            * torch.sqrt(torch.tensor(x_mean.shape[2], dtype=torch.float32))
        )
