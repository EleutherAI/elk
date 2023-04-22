from typing import Literal

import torch
from torch import Tensor, nn


class Normalizer(nn.Module):
    """Basically `BatchNorm` with a less annoying default axis ordering."""

    mean: Tensor
    std: Tensor | None

    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        eps: float = 1e-5,
        mode: Literal["none", "meanonly", "full"] = "full",
    ):
        super().__init__()

        self.eps = eps
        self.mode = mode
        self.normalized_shape = normalized_shape

        self.register_buffer(
            "mean", torch.zeros(*normalized_shape, device=device, dtype=dtype)
        )
        self.register_buffer(
            "std", torch.ones_like(self.mean) if mode == "full" else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Normalize `x` using the stored mean and standard deviation."""
        if self.mode == "none":
            return x
        elif self.std is None:
            return x - self.mean
        else:
            return (x - self.mean) / self.std

    def fit(self, x: Tensor) -> None:
        """Update the stored mean and standard deviation."""

        # Check shape
        num_dims = len(self.normalized_shape)
        if x.shape[-num_dims:] != self.normalized_shape:
            raise ValueError(
                f"Expected trailing sizes {self.normalized_shape} but got "
                f"{x.shape[-num_dims:]}"
            )

        if self.mode == "none":
            return

        dims = [i for i in range(x.ndim - num_dims)]
        if self.std is None:
            torch.mean(x, dim=dims, out=self.mean)
        else:
            variance, self.mean = torch.var_mean(x, dim=dims)
            torch.sqrt(variance + self.eps, out=self.std)
