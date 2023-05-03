"""An ELK reporter network."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import load
from torch import Tensor, nn, optim


@dataclass
class ReporterConfig(ABC, Serializable, decode_into_subclasses=True):
    """
    Args:
        seed: The random seed to use. Defaults to 42.
    """

    seed: int = 42

    @classmethod
    @abstractmethod
    def reporter_class(cls) -> type["Reporter"]:
        """Get the reporter class associated with this config."""


class Reporter(nn.Module, ABC):
    """An ELK reporter network."""

    # Learned Platt scaling parameters
    bias: nn.Parameter
    scale: nn.Parameter

    def reset_parameters(self):
        """Reset the parameters of the probe."""

    @abstractmethod
    def fit(
        self,
        hiddens: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        ...

    @classmethod
    def load(cls, path: Path | str, *, map_location: str = "cpu"):
        """Load a reporter from a file."""
        obj = torch.load(path, map_location=map_location)
        if isinstance(obj, Reporter):  # Backwards compatibility
            return obj

        # Loading a state dict rather than the full object
        elif isinstance(obj, dict):
            cls_path = Path(path).parent / "cfg.yaml"
            cfg = load(ReporterConfig, cls_path)

            # Non-tensor values get passed to the constructor as kwargs
            kwargs = {}
            special_keys = {k for k, v in obj.items() if not isinstance(v, Tensor)}
            for k in special_keys:
                kwargs[k] = obj.pop(k)

            reporter_cls = cfg.reporter_class()
            reporter = reporter_cls(cfg, device=map_location, **kwargs)
            reporter.load_state_dict(obj)
            return reporter
        else:
            raise TypeError(
                f"Expected a `dict` or `Reporter` object, but got {type(obj)}."
            )

    def platt_scale(self, labels: Tensor, hiddens: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            labels: Binary labels of shape [batch].
            hiddens: Hidden states of shape [batch, dim].
            max_iter: Maximum number of iterations for LBFGS.
        """
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(hiddens.dtype).eps,
            tolerance_grad=torch.finfo(hiddens.dtype).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self(hiddens), labels.float()
            )

            loss.backward()
            return float(loss)

        opt.step(closure)
