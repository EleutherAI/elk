"""An ELK reporter network."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from simple_parsing.helpers import Serializable
from torch import Tensor


@dataclass
class ReporterConfig(Serializable):
    """
    Args:
        seed: The random seed to use. Defaults to 42.
    """

    seed: int = 42


@dataclass
class OptimConfig(Serializable):
    """
    Args:
        lr: The learning rate to use. Ignored when `optimizer` is `"lbfgs"`.
            Defaults to 1e-2.
        num_epochs: The number of epochs to train for. Defaults to 1000.
        num_tries: The number of times to try training the reporter. Defaults to 10.
        optimizer: The optimizer to use. Defaults to "adam".
        weight_decay: The weight decay or L2 penalty to use. Defaults to 0.01.
    """

    lr: float = 1e-2
    num_epochs: int = 1000
    num_tries: int = 10
    optimizer: Literal["adam", "lbfgs"] = "lbfgs"
    weight_decay: float = 0.01


class Reporter(nn.Module, ABC):
    """An ELK reporter network."""

    def reset_parameters(self):
        """Reset the parameters of the probe."""

    # TODO: These methods will do something fancier in the future
    @classmethod
    def load(cls, path: Path | str):
        """Load a reporter from a file."""
        return torch.load(path)

    def save(self, path: Path | str):
        # TODO: Save separate JSON and PT files for the reporter.
        torch.save(self, path)

    @abstractmethod
    def fit(
        self,
        hiddens: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        ...
