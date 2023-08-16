from abc import ABC, abstractmethod
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor, nn, optim

from elk.metrics import to_one_hot


class PlattMixin(ABC):
    """Mixin for classifier-like objects that can be Platt scaled."""

    bias: nn.Parameter
    scale: nn.Parameter

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...

    def platt_scale(self, labels: Tensor, hiddens: Tensor, max_iter: int = 100):
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            labels: Binary labels of shape [batch].
            hiddens: Hidden states of shape [batch, dim].
            max_iter: Maximum number of iterations for LBFGS.
        """
        from elk.training import CcsReporter
        norm = None
        try:
            if self.config.norm == "burns":
                norm = "burns"
        except Exception:
            raise ValueError("norm must be specified in config")
        if norm == "burns":
            n, v, k, d = hiddens.shape
            original_hiddens = hiddens
            original_labels = labels
            labels = to_one_hot(repeat(labels, "n -> (n v)", v=v), k).flatten()
            hiddens = rearrange(hiddens, "n v k d -> (n v k) d")

        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(hiddens.dtype).eps,
            tolerance_grad=torch.finfo(hiddens.dtype).eps,
        )

        def closure():
            if norm == "burns":
                opt.zero_grad()
                res = self(original_hiddens)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    rearrange(res, "n v c -> (n v c)"), labels.float()
                )
                loss.backward()
                return float(loss)
            else:
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self(hiddens), labels.float()
                )

                loss.backward()
                return float(loss)

        opt.step(closure)
