"""An ELK reporter network."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, cast

import torch
import torch.nn as nn
from concept_erasure import LeaceFitter
from torch import Tensor

from ..parsing import parse_loss
from ..utils.typing import assert_type
from .common import FitterConfig
from .losses import LOSSES
from .platt_scaling import PlattMixin


@dataclass
class CcsConfig(FitterConfig):
    activation: Literal["gelu", "relu", "swish"] = "gelu"
    """The activation function to use."""
    bias: bool = True
    """Whether to use a bias term in the linear layers."""
    hidden_size: Optional[int] = None
    """
    The number of hidden units in the MLP. Defaults to None. By default, use an MLP
    expansion ratio of 4/3. This ratio is used by Tucker et al. (2022)
    <https://arxiv.org/abs/2204.09722> in their 3-layer MLP probes. We could also use
    a ratio of 4, imitating transformer FFNs, but this seems to lead to excessively
    large MLPs when num_layers > 2.
    """
    init: Literal["default", "pca", "spherical", "zero"] = "default"
    """The initialization scheme to use."""
    loss: list[str] = field(default_factory=lambda: ["ccs"])
    """
    The loss function to use. list of strings, each of the form "coef*name", where coef
    is a float and name is one of the keys in `elk.training.losses.LOSSES`.
    Example: `--loss 1.0*consistency_squared 0.5*prompt_var` corresponds to the loss
    function 1.0*consistency_squared + 0.5*prompt_var.
    """
    loss_dict: dict[str, float] = field(default_factory=dict, init=False)
    num_layers: int = 1
    """The number of layers in the MLP."""
    pre_ln: bool = False
    """Whether to include a LayerNorm module before the first linear layer."""
    supervised_weight: float = 0.0
    """The weight of the supervised loss."""

    lr: float = 1e-2
    """The learning rate to use. Ignored when `optimizer` is `"lbfgs"`."""
    num_epochs: int = 1000
    """The number of epochs to train for."""
    num_tries: int = 10
    """The number of times to try training the reporter."""
    optimizer: Literal["adam", "lbfgs"] = "lbfgs"
    """The optimizer to use."""
    weight_decay: float = 0.01
    """The weight decay or L2 penalty to use."""

    def __post_init__(self):
        self.loss_dict = parse_loss(self.loss)

        # standardize the loss field
        self.loss = [f"{coef}*{name}" for name, coef in self.loss_dict.items()]


class CcsReporter(nn.Module, PlattMixin):
    """CCS reporter network.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    config: CcsConfig

    def __init__(
        self,
        cfg: CcsConfig,
        in_features: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_variants: int = 1,
    ):
        super().__init__()
        self.config = cfg
        self.in_features = in_features
        self.num_variants = num_variants

        # Learnable Platt scaling parameters
        self.bias = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        hidden_size = cfg.hidden_size or 4 * in_features // 3

        self.norm = None
        self.probe = nn.Sequential(
            nn.Linear(
                in_features,
                1 if cfg.num_layers < 2 else hidden_size,
                bias=cfg.bias,
                device=device,
            ),
        )
        if cfg.pre_ln:
            self.probe.insert(0, nn.LayerNorm(in_features, elementwise_affine=False))

        act_cls = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "swish": nn.SiLU,
        }[cfg.activation]

        for i in range(1, cfg.num_layers):
            self.probe.append(act_cls())
            self.probe.append(
                nn.Linear(
                    hidden_size,
                    1 if i == cfg.num_layers - 1 else hidden_size,
                    bias=cfg.bias,
                    device=device,
                )
            )

    def reset_parameters(self):
        """Reset the parameters of the probe.

        If init is "spherical", use the spherical initialization scheme.
        If init is "default", use the default PyTorch initialization scheme for
        nn.Linear (Kaiming uniform).
        If init is "zero", initialize all parameters to zero.
        """
        if self.config.init == "spherical":
            # Mathematically equivalent to the unusual initialization scheme used in
            # the original paper. They sample a Gaussian vector of dim in_features + 1,
            # normalize to the unit sphere, then add an extra all-ones dimension to the
            # input and compute the inner product. Here, we use nn.Linear with an
            # explicit bias term, but use the same initialization.
            assert len(self.probe) == 1, "Only linear probes can use spherical init"
            probe = cast(nn.Linear, self.probe[0])  # Pylance gets the type wrong here

            theta = torch.randn(1, probe.in_features + 1, device=probe.weight.device)
            theta /= theta.norm()
            probe.weight.data = theta[:, :-1]
            probe.bias.data = theta[:, -1]

        elif self.config.init == "default":
            for layer in self.probe:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

        elif self.config.init == "zero":
            for param in self.parameters():
                param.data.zero_()
        elif self.config.init != "pca":
            raise ValueError(f"Unknown init: {self.config.init}")

    def forward(self, x: Tensor) -> Tensor:
        """Return the credence assigned to the hidden state `x`."""
        assert self.norm is not None, "Must call fit() before forward()"

        raw_scores = self.probe(self.norm(x)).squeeze(-1)
        return raw_scores.mul(self.scale).add(self.bias).squeeze(-1)

    def loss(self, logit0: Tensor, logit1: Tensor) -> Tensor:
        """Return the loss of the reporter on the contrast pair (x0, x1).

        Args:
            logit0: The raw score output of the reporter on x0.
            logit1: The raw score output of the reporter on x1.

        Returns:
            loss: The loss of the reporter on the contrast pair (x0, x1).
        """
        loss = sum(
            LOSSES[name](logit0, logit1, coef)
            for name, coef in self.config.loss_dict.items()
        )
        return assert_type(Tensor, loss)

    def fit(self, hiddens: Tensor) -> float:
        """Fit the probe to the contrast pair `hiddens`.

        Returns:
            best_loss: The best loss obtained.
        """
        x_neg, x_pos = hiddens.unbind(2)

        # One-hot indicators for each prompt template
        n, v, d = x_neg.shape
        prompt_ids = torch.eye(v, device=x_neg.device).expand(n, -1, -1)

        fitter = LeaceFitter(d, 2 * v, dtype=x_neg.dtype, device=x_neg.device)
        fitter.update(
            x=x_neg,
            # Independent indicator for each (template, pseudo-label) pair
            z=torch.cat([torch.zeros_like(prompt_ids), prompt_ids], dim=-1),
        )
        fitter.update(
            x=x_pos,
            # Independent indicator for each (template, pseudo-label) pair
            z=torch.cat([prompt_ids, torch.zeros_like(prompt_ids)], dim=-1),
        )
        self.norm = fitter.eraser

        x_neg, x_pos = self.norm(x_neg), self.norm(x_pos)

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, Tensor] = {}  # State dict of the best run

        for i in range(self.config.num_tries):
            self.reset_parameters()

            # This is sort of inefficient but whatever
            if self.config.init == "pca":
                diffs = torch.flatten(x_pos - x_neg, 0, 1)
                _, __, V = torch.pca_lowrank(diffs, q=i + 1)
                self.probe[0].weight.data = V[:, -1, None].T

            if self.config.optimizer == "lbfgs":
                loss = self.train_loop_lbfgs(x_neg, x_pos)
            elif self.config.optimizer == "adam":
                loss = self.train_loop_adam(x_neg, x_pos)
            else:
                raise ValueError(f"Optimizer {self.config.optimizer} is not supported")

            if loss < best_loss:
                best_loss = loss
                best_state = deepcopy(self.state_dict())

        if not math.isfinite(best_loss):
            raise RuntimeError("Got NaN/infinite loss during training")

        self.load_state_dict(best_state)
        return best_loss

    def train_loop_adam(self, x_neg: Tensor, x_pos: Tensor) -> float:
        """Adam train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        loss = torch.inf
        for _ in range(self.config.num_epochs):
            optimizer.zero_grad()

            # We already normalized in fit()
            loss = self.loss(self(x_neg), self(x_pos))
            loss.backward()
            optimizer.step()

        return float(loss)

    def train_loop_lbfgs(self, x_neg: Tensor, x_pos: Tensor) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.config.num_epochs,
            tolerance_change=torch.finfo(x_pos.dtype).eps,
            tolerance_grad=torch.finfo(x_pos.dtype).eps,
        )
        # Raw unsupervised loss, WITHOUT regularization
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # We already normalized in fit()
            loss = self.loss(self(x_neg), self(x_pos))
            regularizer = 0.0

            # We explicitly add L2 regularization to the loss, since LBFGS
            # doesn't have a weight_decay parameter
            for param in self.parameters():
                regularizer += self.config.weight_decay * param.norm() ** 2 / 2

            regularized = loss + regularizer
            regularized.backward()

            return float(regularized)

        optimizer.step(closure)
        return float(loss)
