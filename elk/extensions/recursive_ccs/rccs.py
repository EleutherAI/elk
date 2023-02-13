from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from elk.training.ccs import CCS
from torch import nn
import torch
from torch import Tensor
from itertools import count
from elk.training.ccs import EvalResult
import torch.nn.utils.parametrize as P


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

        ccs = CCS(
            in_features=in_features,
            **ccs_params,
        )
        if parametrization is not None:
            P.register_parametrization(ccs.probe[0], "weight", parametrization)

        train_loss = ccs.fit(data, **train_params)
        self.probes.append(ccs)
        return ccs, train_loss

    def score(self, data: tuple[Tensor, Tensor], labels: Tensor) -> list[EvalResult]:
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

    @classmethod
    def save(cls, path: Path, *rccs: "RecursiveCCS") -> None:
        """Saves the probes of the RecursiveCCS to the given path.

        Remove the parametrizations, because they are not serializable."""
        for r in rccs:
            for probe in r.probes:
                # if has parametrization, remove them
                if hasattr(probe.probe[0], "parametrizations"):
                    P.remove_parametrizations(probe.probe[0], "weight")
        torch.save(rccs, path)

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> list["RecursiveCCS"]:
        """Loads the probes from the given path.

        Recover the parametrizations."""
        rccs = torch.load(path)
        # Recover parametrizations
        for r in rccs:
            new_r = RecursiveCCS(device=device)
            for probe in r.probes:
                new_r.probes.append(probe)
                parametrization = new_r.get_next_parametrization()
                if parametrization is not None:
                    P.register_parametrization(
                        probe.probe[0], "weight", parametrization
                    )


def project(x: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Projects on the hyperplane defined by the constraints.

    After the projection, <x, constraints[i]> = 0 for all i."""
    inner_products = torch.einsum("...h,nh->...n", x, constraints)
    return x - torch.einsum("...n,nh->...h", inner_products, constraints)


def assert_orthogonal(directions: torch.Tensor, atol: float = 1e-6) -> None:
    """Asserts that the directions are orthogonal."""
    inner_products = torch.einsum("nh,mh->nm", directions, directions)

    diff = torch.abs(
        inner_products
        - torch.eye(
            len(directions), dtype=inner_products.dtype, device=inner_products.device
        )
    ).max()
    assert diff < atol, f"Directions are not orthogonal: {diff=}"
