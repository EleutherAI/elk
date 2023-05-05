import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpectralNorm(nn.Module):
    """Removes the subspace responsible for correlations between hiddens and labels."""

    mean_x: Tensor
    """Running mean of X."""

    mean_y: Tensor
    """Running mean of Y."""

    u: Tensor
    """Orthonormal basis of the subspace to remove."""

    x_M2: Tensor
    """Unnormalized second moment of X."""

    y_M2: Tensor
    """Unnormalized second moment of Y."""

    xcov_M2: Tensor
    """Unnormalized cross-covariance matrix X^T Y."""

    n: Tensor
    """Number of samples seen so far."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        standardize: bool = False,
    ):
        super().__init__()

        self.register_buffer(
            "mean_x", torch.zeros(num_features, device=device, dtype=dtype)
        )
        self.register_buffer(
            "mean_y", torch.zeros(num_classes, device=device, dtype=dtype)
        )
        self.register_buffer(
            "u", torch.zeros(num_features, num_classes, device=device, dtype=dtype)
        )
        self.register_buffer(
            "x_M2", torch.zeros(num_features, device=device, dtype=dtype)
        )
        self.register_buffer(
            "xcov_M2",
            torch.zeros(num_features, num_classes, device=device, dtype=dtype),
        )
        self.register_buffer(
            "y_M2", torch.zeros(num_classes, device=device, dtype=dtype)
        )
        self.register_buffer("n", torch.tensor(0, device=device, dtype=dtype))
        self.standardize = standardize

    def forward(self, x: Tensor) -> Tensor:
        """Remove the subspace responsible for correlations between x and y."""
        d, _ = self.xcov_M2.shape
        assert self.n > 0, "Call update() before forward()"
        assert x.shape[-1] == d

        # First center the input
        x_ = x - self.mean_x
        if self.standardize:
            x_ /= torch.sqrt(self.var_x + 1e-5)

        # Remove the subspace
        x_ -= (x_ @ self.u) @ self.u.mT

        return x_

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor) -> "SpectralNorm":
        """Update the running statistics with a new batch of data."""
        x = x.flatten(0, -2).type_as(self.mean_x)

        n, d = x.shape
        d2, c = self.xcov_M2.shape
        assert d == d2, "Unexpected number of features"

        # y might start out 1D, but we want to treat it as 2D
        y = y.reshape(n, -1).type_as(x)
        assert y.shape[-1] == c, "Unexpected number of classes"

        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n
        delta_x2 = x - self.mean_x

        delta_y = y - self.mean_y
        self.mean_y += delta_y.sum(dim=0) / self.n
        delta_y2 = y - self.mean_y

        self.x_M2 += torch.sum(delta_x * delta_x2, dim=0)
        self.y_M2 += torch.sum(delta_y * delta_y2, dim=0)
        self.xcov_M2.addmm_(delta_x.mT, delta_y2)

        # If we're using one-hot encoded binary labels, we can compute the
        # projection matrix without SVD
        mat = self.xcorr if self.standardize else self.xcov
        if c == 1:
            self.u = F.normalize(mat, dim=0)
        else:
            self.u, _, __ = torch.svd_lowrank(mat, q=c)

        return self

    @property
    def P(self) -> Tensor:
        """Projection matrix for removing the subspace."""
        eye = torch.eye(self.u.shape[0], device=self.u.device, dtype=self.u.dtype)
        return eye - self.u @ self.u.mT

    @property
    def var_x(self) -> Tensor:
        """The variance of X."""
        return self.x_M2 / self.n

    @property
    def var_y(self) -> Tensor:
        """The variance of Y."""
        return self.y_M2 / self.n

    @property
    def xcov(self) -> Tensor:
        """The cross-covariance matrix."""
        return self.xcov_M2 / self.n

    @property
    def xcorr(self) -> Tensor:
        """The cross-correlation matrix."""
        return self.xcov / torch.sqrt(self.var_x[:, None] * self.var_y[None, :] + 1e-5)
