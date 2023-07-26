import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization  Minimal changes from the original code."""

    def forward(self, x_all: Tensor) -> Tensor:
        res = []
        xs = x_all.unbind(dim=1)
        for x in xs:
            x_mean: Tensor = x - x.mean(dim=0)
            std = torch.linalg.norm(x_mean, axis=0) / torch.sqrt(
                torch.tensor(x_mean.shape[0], dtype=torch.float32)
            )
            avg_norm = std.mean()
            res.append(x_mean / avg_norm)
        return torch.stack(res, dim=1)
