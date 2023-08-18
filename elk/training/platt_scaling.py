from abc import ABC, abstractmethod
from typing import Any

import torch
from einops import rearrange, repeat
from rich import print
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

        n, v, k, d = hiddens.shape
        original_hiddens = hiddens
        squashed_labels = to_one_hot(repeat(labels, "n -> (n v)", v=v), k).flatten()
        squashed_hiddens = rearrange(hiddens, "n v k d -> (n v k) d")

        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(squashed_hiddens.dtype).eps,
            tolerance_grad=torch.finfo(squashed_hiddens.dtype).eps,
        )

        losses = []

        def closure():
            opt.zero_grad()
            if self.config.platt_burns == "hack":
                res = self(original_hiddens).flatten()
            else:
                res = self(squashed_hiddens)
            loss = nn.functional.binary_cross_entropy_with_logits(
                res, squashed_labels.float()
            )
            loss.backward()
            losses.append(float(loss))
            return float(loss)

        opt.step(closure)


        print("platt losses", losses)
        print("scale", self.scale.item())
        print("bias", self.bias.item())
