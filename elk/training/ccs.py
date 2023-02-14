from .losses import ccs_squared_loss, js_loss
from ..utils import maybe_ddp_wrap, maybe_all_gather, maybe_all_reduce
from copy import deepcopy
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy as bce
from typing import cast, Literal, NamedTuple, Optional, Type, Union
import math
import torch
import torch.nn as nn


class EvalResult(NamedTuple):
    """The result of evaluating a CCS model on a dataset."""

    loss: float
    acc: float
    cal_acc: float
    auroc: float


class CCS(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        activation: Type[nn.Module] = nn.GELU,
        bias: bool = True,
        device: Optional[str] = None,
        hidden_size: Optional[int] = None,
        init: Literal["default", "spherical", "zero"] = "zero",
        loss: Literal["js", "squared"] = "squared",
        num_layers: int = 1,
        pre_ln: bool = False,
        supervised_weight: float = 0.0,
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
        if pre_ln:
            # Include a LayerNorm module before the first linear layer
            self.probe.insert(0, nn.LayerNorm(in_features, elementwise_affine=False))

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
        self.unsupervised_loss = js_loss if loss == "js" else ccs_squared_loss
        self.supervised_weight = supervised_weight

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
        elif self.init == "zero":
            for param in self.parameters():
                param.data.zero_()
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

    def loss(
        self,
        logit0: torch.Tensor,
        logit1: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the loss of the probe on the contrast pair `(x0, x1)`."""
        loss = self.unsupervised_loss(logit0, logit1)

        # If labels are provided, use them to compute a supervised loss
        if labels is not None:
            num_labels = len(labels)
            assert num_labels <= len(logit0), "Too many labels provided"
            p0 = logit0[:num_labels].sigmoid()
            p1 = logit1[:num_labels].sigmoid()

            alpha = self.supervised_weight
            preds = p0.add(1 - p1).mul(0.5).squeeze(-1)
            bce_loss = bce(preds, labels.type_as(preds))
            loss = alpha * bce_loss + (1 - alpha) * loss

        elif self.supervised_weight > 0:
            raise ValueError(
                "Supervised weight > 0 but no labels provided to compute loss"
            )

        return loss

    def validate_data(self, data):
        assert len(data) == 2 and data[0].shape == data[1].shape

    def fit(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        *,
        lr: float = 1e-2,
        num_epochs: int = 1000,
        num_tries: int = 10,
        optimizer: Literal["adam", "lbfgs"] = "adam",
        verbose: bool = False,
        weight_decay: float = 0.01,
    ) -> float:
        self.validate_data(contrast_pair)
        if verbose:
            print(f"Fitting CCS probe; {num_epochs=}, {num_tries=}, {lr=}")

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, torch.Tensor] = {}  # State dict of the best run
        x0, x1 = contrast_pair

        for _ in range(num_tries):
            self.reset_parameters()

            if optimizer == "lbfgs":
                loss = self.train_loop_lbfgs(x0, x1, labels, num_epochs, weight_decay)
            elif optimizer == "adam":
                loss = self.train_loop_adam(
                    x0, x1, labels, lr, num_epochs, weight_decay
                )
            else:
                raise ValueError(f"Optimizer {optimizer} is not supported")

            if loss < best_loss:
                if verbose:
                    print(f"Found new best params; loss: {loss:.4f}")

                best_loss = loss
                best_state = deepcopy(self.state_dict())

        if not math.isfinite(best_loss):
            raise RuntimeError("Got NaN/infinite loss during training")

        self.load_state_dict(best_state)
        return best_loss

    @torch.no_grad()
    def score(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> EvalResult:
        self.validate_data(contrast_pair)

        logit0, logit1 = map(self, contrast_pair)
        p0, p1 = logit0.sigmoid(), logit1.sigmoid()
        pred_probs = 0.5 * (p0 + (1 - p1))

        pred_probs = maybe_all_gather(pred_probs)
        labels = maybe_all_gather(labels)

        # Calibrated accuracy
        cal_thresh = pred_probs.float().quantile(labels.float().mean())
        cal_preds = pred_probs.gt(cal_thresh).squeeze(1).to(torch.int)
        raw_preds = pred_probs.gt(0.5).squeeze(1).to(torch.int)

        auroc = float(roc_auc_score(labels.cpu(), pred_probs.cpu()))
        cal_acc = cal_preds.eq(labels.reshape(-1)).float().mean()
        raw_acc = raw_preds.eq(labels.reshape(-1)).float().mean()

        return EvalResult(
            loss=self.loss(logit0, logit1).item(),
            acc=torch.max(raw_acc, 1 - raw_acc).item(),
            cal_acc=torch.max(cal_acc, 1 - cal_acc).item(),
            auroc=max(auroc, 1 - auroc),
        )

    def train_loop_adam(
        self,
        x0,
        x1,
        labels: Optional[torch.Tensor],
        lr: float,
        num_epochs: int,
        wd: float,
    ) -> float:
        """Adam train loop, returning the final loss. Modifies params in-place."""

        probe = maybe_ddp_wrap(self)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

        loss = torch.inf
        for _ in range(num_epochs):
            optimizer.zero_grad()

            loss = self.loss(probe(x0), probe(x1), labels)
            loss.backward()
            optimizer.step()

        return float(loss)

    def train_loop_lbfgs(
        self, x0, x1, labels: Optional[torch.Tensor], max_iter: int, l2_penalty: float
    ) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(x0.dtype).eps,
            tolerance_grad=torch.finfo(x0.dtype).eps,
        )
        # Raw unsupervised loss, WITHOUT regularization
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            loss = self.loss(self(x0), self(x1), labels)
            regularizer = 0.0

            # We explicitly add L2 regularization to the loss, since LBFGS
            # doesn't have a weight_decay parameter
            for param in self.parameters():
                regularizer += l2_penalty * param.norm() ** 2 / 2

            regularized = loss + regularizer
            regularized.backward()

            for p in self.parameters():
                if p.grad is not None:
                    maybe_all_reduce(p.grad)

            return float(regularized)

        optimizer.step(closure)
        return float(loss)
