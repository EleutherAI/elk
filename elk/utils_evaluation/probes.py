import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


# abstract probe class
class Probe(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def normalize(self):
        """Done at the end of each training step.
        Can return a loss to be added to the loss."""
        return 0


class LinearProbe(Probe):
    def __init__(self, input_dim, include_bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=include_bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class LinearOrthogonalProbe(Probe):
    def __init__(self, orthogonal_constraints):
        """kernel weights will orthogonal to each orthogonal_constraint
        along_directions of shape (n_directions, input_dim)"""
        super().__init__()
        _, d = orthogonal_constraints.shape
        self.linear = nn.Linear(d, 1)
        self.orthogonal_constraints = orthogonal_constraints

    def forward(self, x):
        w = project(self.linear.weight, self.orthogonal_constraints)
        y = F.linear(x, w, self.linear.bias)
        return torch.sigmoid(y)

    def normalize(self):
        with torch.no_grad():
            self.linear.weight[:] = project(
                self.linear.weight, self.orthogonal_constraints
            )
        return 0


class LinearAlongProbe(Probe):
    def __init__(self, along_directions):
        """kernel weights will be along the along_directions
        along_directions of shape (n_directions, input_dim)"""
        super().__init__()
        n, _ = along_directions.shape
        self.coeffs = nn.Linear(n, 1)
        self.coeffs.weight.data = torch.ones_like(self.coeffs.weight.data) / n
        self.along_directions = along_directions

    def forward(self, x):
        y = F.linear(x, self.weight, self.coeffs.bias)
        return torch.sigmoid(y)

    @property
    def weight(self):
        return torch.einsum("nd,nh->hd", self.along_directions, self.coeffs.weight)


class MLPProbe(Probe):
    def __init__(self, *layer_sizes) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            if i != 0:
                self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


def project(x: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Projects on the hyperplane defined by the constraints"""
    inner_products = torch.einsum("...h,nh->...n", x, constraints)
    return x - torch.einsum("...n,nh->...h", inner_products, constraints)
