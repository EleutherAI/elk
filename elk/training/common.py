"""An ELK reporter network."""

from dataclasses import dataclass

from concept_erasure import LeaceEraser
from simple_parsing.helpers import Serializable
from torch import Tensor, nn

from .platt_scaling import PlattMixin


@dataclass
class FitterConfig(Serializable, decode_into_subclasses=True):
    seed: int = 42
    """The random seed to use."""


@dataclass
class Reporter(PlattMixin):
    weight: Tensor
    eraser: LeaceEraser

    def __post_init__(self):
        # Platt scaling parameters
        self.bias = nn.Parameter(self.weight.new_zeros(1))
        self.scale = nn.Parameter(self.weight.new_ones(1))

    def __call__(self, hiddens: Tensor) -> Tensor:
        """Return the predicted log odds on input `x`."""
        raw_scores = self.eraser(hiddens) @ self.weight.mT
        return raw_scores.mul(self.scale).add(self.bias).squeeze(-1)
