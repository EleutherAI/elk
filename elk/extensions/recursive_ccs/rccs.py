from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from elk.training.ccs import CCS
from torch import nn
import torch
from torch import Tensor
from itertools import count


@dataclass
class RecursiveCCS:
    probes: list[CCS] = field(default_factory=list)
    device: str = "cuda"

    def fit_next_probe(
        self, data: tuple[Tensor, Tensor], ccs_params, train_params
    ) -> tuple[CCS, float]:
        """Finds the next probe by training a new probe and comparing it to the
        current probes."""
        parametrization = self.get_next_parametrization()
        in_features = data[0].shape[1]

        probe = CCS(
            in_features=in_features,
            first_linear_parametrization=parametrization,
            **ccs_params,
        )
        train_loss = probe.fit(data, **train_params)
        self.probes.append(probe)
        return probe, train_loss

    def score_all(self, data: tuple[Tensor, Tensor], labels: Tensor) -> list[float]:
        """Scores all probes."""
        return [probe.score(data, labels) for probe in self.probes]

    def get_directions(self) -> Tensor:
        """Returns the directions of the current probes."""
        directions = torch.cat(
            [probe.probe[0].weight.detach() for probe in self.probes]
        )
        directions /= torch.norm(directions, dim=1, keepdim=True)
        return directions

    def get_next_parametrization(self) -> Optional[nn.Module]:
        """Returns a parametrization for the next probe.

        The parametrization projects on the hyperplane orthogonal to
        the directions of the current probes."""

        if not self.probes:
            return None

        directions = self.get_directions()
        assert_orthogonal(directions)

        class OrthogonalProjection(nn.Module):
            def __init__(self, constraints):
                super().__init__()
                self.constraints = constraints

            # no right_inverse here, because we parametrize by the tensor itself

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return project(x, self.constraints)

        return OrthogonalProjection(directions)


def project(x: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Projects on the hyperplane defined by the constraints.

    After the projection, <x, constraints[i]> = 0 for all i."""
    inner_products = torch.einsum("...h,nh->...n", x, constraints)
    return x - torch.einsum("...n,nh->...h", inner_products, constraints)


def assert_orthogonal(directions: torch.Tensor, atol: float = 1e-6) -> None:
    """Asserts that the directions are orthogonal."""
    inner_products = torch.einsum("nh,mh->nm", directions, directions)
    assert torch.allclose(
        inner_products, torch.eye(len(directions)).to(directions.device), atol=atol
    )
