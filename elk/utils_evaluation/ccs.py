from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from .losses import ccs_squared_loss
from typing import cast, Literal, Optional, Type, Union
import torch
import torch.nn as nn


@dataclass
class TrainParams:
    num_epochs: int = 1000
    num_tries: int = 10
    lr: float = 1e-2
    weight_decay: float = 0.01
    optimizer: Literal["adam", "lbfgs"] = "adam"


class CCS(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        activation: Type[nn.Module] = nn.GELU,
        bias: bool = True,
        device: str = "cuda",
        hidden_size: Optional[int] = None,
        init: Literal["default", "spherical"] = "default",
        num_layers: int = 1,
    ):
        super().__init__()

        # By default, use an MLP expansion ratio of 4/3. This ratio is used by
        # Tucker et al. (2022) <https://arxiv.org/abs/2204.09722> in their 3-layer
        # MLP probes. We could also use a ratio of 4, imitating transformer FFNs,
        # but this seems to lead to excessively large MLPs when num_layers > 2.
        hidden_size = hidden_size or 4 * in_features // 3

        self.probe = nn.Sequential(
            nn.Linear(
                in_features,
                1 if num_layers < 2 else hidden_size,
                bias=bias,
                device=device,
            ),
        )

        for i in range(1, num_layers):
            self.probe.append(activation())
            self.probe.append(
                nn.Linear(
                    hidden_size,
                    1 if i == num_layers - 1 else hidden_size,
                    bias=bias,
                    device=device,
                )
            )

        self.init = init
        self.device = device

    def reset_parameters(self):
        # Mathematically equivalent to the unusual initialization scheme used in the
        # original paper. They sample a random Gaussian vector of dim in_features + 1,
        # normalize to the unit sphere, then add an extra all-ones dimension to the
        # input and compute the inner product. Here, we use nn.Linear with an explicit
        # bias term, but use the same initialization.
        if self.init == "spherical":
            assert len(self.probe) == 1, "Only linear probes can use spherical init"
            probe = cast(nn.Linear, self.probe[0])  # Pylance gets the type wrong here

            theta = torch.randn(1, probe.in_features + 1, device=probe.weight.device)
            theta /= theta.norm()
            probe.weight.data = theta[:, :-1]
            probe.bias.data = theta[:, -1]

        # Default PyTorch initialization (Kaiming uniform)
        elif self.init == "default":
            for layer in self.probe:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        else:
            raise ValueError(f"Unknown init: {self.init}")

    # These methods will do something fancier in the future
    @classmethod
    def load(cls, path: Union[Path, str]):
        return torch.load(path)

    def save(self, path: Union[Path, str]):
        # TODO: Save separate JSON and PT files for the CCS model.
        torch.save(self, path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.probe(x)

    def fit_once(self, x0, x1, train_params: TrainParams):
        """
        Do a single training run of num_epochs epochs
        """
        if train_params.optimizer == "lbfgs":
            loss = self.train_loop_lbfgs(x0, x1, train_params)
        elif train_params.optimizer == "adam":
            loss = self.train_loop_adam(x0, x1, train_params)
        else:
            raise ValueError(f"Optimizer {train_params.optimizer} is not supported")

        return float(loss)

    def validate_data(self, data):
        assert len(data) == 2 and data[0].shape == data[1].shape

    def fit(
        self,
        data: tuple[torch.Tensor, torch.Tensor],
        lr: float = 1e-2,
        num_epochs: int = 1000,
        num_tries: int = 10,
        optimizer: Literal["adam", "lbfgs"] = "adam",
        verbose: bool = False,
        weight_decay: float = 0.01,
    ):
        self.validate_data(data)
        if verbose:
            print(f"Fitting CCS probe; {num_epochs=}, {num_tries=}, {lr=}")

        train_params = TrainParams(
            num_epochs=num_epochs,
            num_tries=num_tries,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
        )

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, torch.Tensor] = {}  # State dict of the best run
        x0, x1 = data

        for _ in range(train_params.num_tries):
            self.reset_parameters()

            loss = self.fit_once(x0, x1, train_params)
            if loss < best_loss:
                if verbose:
                    print(f"Found new best params; loss: {loss:.4f}")

                best_loss = loss
                best_state = deepcopy(self.probe.state_dict())

        self.probe.load_state_dict(best_state)

    @torch.no_grad()
    def score(
        self, data: tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor
    ) -> tuple[float, float]:
        self.validate_data(data)

        logit0, logit1 = map(self, data)
        p0, p1 = logit0.sigmoid(), logit1.sigmoid()
        avg_pred = 0.5 * (p0 + (1 - p1))

        predictions = avg_pred.lt(0.5).squeeze(1).to(int)
        raw_acc = predictions.eq(labels.reshape(-1)).float().mean()
        max_acc = torch.max(raw_acc, 1 - raw_acc).item()

        return max_acc, ccs_squared_loss(logit0, logit1).item()

    def train_loop_adam(self, x0, x1, train_params: TrainParams) -> float:
        """Adam train loop, returning the final loss. Modifies params in-place."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
        )

        loss = torch.inf
        for _ in range(train_params.num_epochs):
            optimizer.zero_grad()

            logit0, logit1 = self(x0), self(x1)
            loss = ccs_squared_loss(logit0, logit1)

            loss.backward()
            optimizer.step()

        return float(loss)

    def train_loop_lbfgs(self, x0, x1, train_params: TrainParams) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=train_params.num_epochs,
            tolerance_change=torch.finfo(x0.dtype).eps,
            tolerance_grad=torch.finfo(x0.dtype).eps,
        )
        # Raw unsupervised loss, WITHOUT regularization
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            logit0, logit1 = self(x0), self(x1)
            loss = ccs_squared_loss(logit0, logit1)
            regularizer = 0.0

            # We explicitly add L2 regularization to the loss, since LBFGS
            # doesn't have a weight_decay parameter
            for param in self.parameters():
                regularizer += train_params.weight_decay * param.norm() ** 2 / 2

            regularized = loss + regularizer
            regularized.backward()

            return float(regularized)

        optimizer.step(closure)
        return float(loss)
