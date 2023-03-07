from .reporter import OptimConfig, Reporter, ReporterConfig
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
import math
import torch
import torch.nn as nn


@dataclass
class MlpReporterConfig(ReporterConfig):
    """
    Args:
        activation: The activation function to use. Defaults to GELU.
        bias: Whether to use a bias term in the linear layers. Defaults to True.
        hidden_size: The number of hidden units in the MLP. Defaults to None.
            By default, use an MLP expansion ratio of 4/3. This ratio is used by
            Tucker et al. (2022) <https://arxiv.org/abs/2204.09722> in their 3-layer
            MLP probes. We could also use a ratio of 4, imitating transformer FFNs,
            but this seems to lead to excessively large MLPs when num_layers > 2.
        init: The initialization scheme to use. Defaults to "zero".
        num_layers: The number of layers in the MLP. Defaults to 1.
    """

    activation: Literal["gelu", "relu", "swish"] = "gelu"
    bias: bool = True
    hidden_size: Optional[int] = None
    init: Literal["default", "spherical", "zero"] = "default"
    num_layers: int = 2


class MlpReporter(Reporter):
    config_cls = MlpReporterConfig
    config: MlpReporterConfig

    def __init__(
        self, in_features: int, cfg: MlpReporterConfig, device: Optional[str] = None
    ):
        super().__init__(in_features, cfg, device)

        hidden_size = cfg.hidden_size or 4 * in_features // 3

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features,
                1 if cfg.num_layers < 2 else hidden_size,
                bias=cfg.bias,
                device=device,
            ),
        )

        act_cls = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "swish": nn.SiLU,
        }[cfg.activation]

        for i in range(1, cfg.num_layers):
            self.mlp.append(act_cls())
            self.mlp.append(
                nn.Linear(
                    hidden_size,
                    1 if i == cfg.num_layers - 1 else hidden_size,
                    bias=cfg.bias,
                    device=device,
                )
            )

    def reset_parameters(self):
        """Reset the parameters of the probe."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.mlp(x)

    def fit(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        cfg: OptimConfig = OptimConfig(),
    ) -> float:
        self.validate_data(contrast_pair)

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, torch.Tensor] = {}  # State dict of the best run
        x0, x1 = contrast_pair

        for _ in range(self.config.num_heads):
            self.reset_parameters()

            if cfg.optimizer == "lbfgs":
                loss = self.fit_once_lbfgs(x0, x1, labels, cfg)
            elif cfg.optimizer == "adam":
                loss = self.train_loop_adam(x0, x1, labels, cfg)
            else:
                raise ValueError(f"Optimizer {cfg.optimizer} is not supported")

            if loss < best_loss:
                best_loss = loss
                best_state = deepcopy(self.state_dict())

        if not math.isfinite(best_loss):
            raise RuntimeError("Got NaN/infinite loss during training")

        self.load_state_dict(best_state)
        return best_loss
