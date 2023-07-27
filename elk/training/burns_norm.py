import torch
from torch import Tensor, nn


class BurnsNorm(nn.Module):
    """Burns et al. style normalization. Minimal changes from the original code."""

    # TODO: Clean this up... + write test
    def forward(self, x_all: Tensor) -> Tensor:
        assert len(x_all.shape) in [3, 4]
        if x_all.dim() == 3:
            return self.three_dims(x_all)
        else:
            return self.four_dims(x_all)

    def three_dims(self, x_all: Tensor) -> Tensor:
        """Normalize the input tensor.
        x_all: Tensor of shape (n, v, d)
        """
        x_normalized: Tensor = x_all - x_all.mean(dim=0)

        std = torch.linalg.norm(x_normalized, axis=0) / torch.sqrt(
            torch.tensor(x_normalized.shape[0], dtype=torch.float32)
        )
        assert len(std.shape) == 2
        # We want to mean over everything except the v dimension, which is dim=0
        # dim=0 is v, since after doing torch.linalg.norm we end up with a tensor
        # missing the first dimension n
        avg_norm = std.mean(dim=(1))

        # add singleton dimension at beginnign and end to allow broadcasting
        return x_normalized / avg_norm.unsqueeze(0).unsqueeze(-1)

    def four_dims(self, x_all: Tensor) -> Tensor:
        """Normalize the input tensor.
        x_all: Tensor of shape (n, v, k, d)
        """
        x_normalized: Tensor = x_all - x_all.mean(dim=0)

        std = torch.linalg.norm(x_normalized, axis=0) / torch.sqrt(
            torch.tensor(x_normalized.shape[0], dtype=torch.float32)
        )
        assert len(std.shape) == 3
        # We want to mean over everything except the v dimension, which is dim=0
        # dim=0 is v, since after doing torch.linalg.norm we end up with a tensor
        # missing the first dimension n
        avg_norm = std.mean(dim=(1, 2))

        # add singleton dimension at beginning and end to allow broadcasting
        return x_normalized / avg_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def correct_but_slow_normalization(self, x_all: Tensor) -> Tensor:
        # TODO: Remove this once everything is cleandup
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
