"""An ELK reporter network."""

from .losses import ccs_squared_loss, js_loss
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import roc_auc_score
from simple_parsing.helpers import Serializable
from torch.nn.functional import binary_cross_entropy as bce
from typing import cast, Literal, NamedTuple, Optional, Union
import math
import torch
import torch.nn as nn


class EvalResult(NamedTuple):
    """The result of evaluating a reporter on a dataset.

    The `.score()` function of a reporter returns an instance of this class,
    which contains the loss, accuracy, calibrated accuracy, and AUROC.
    """

    loss: float
    acc: float
    cal_acc: float
    auroc: float


@dataclass
class ReporterConfig(Serializable):
    """
    Args:
        activation: The activation function to use. Defaults to GELU.
        bias: Whether to use a bias term in the linear layers. Defaults to True.
        device: The device to use. Defaults to None, which means "current device".
        hidden_size: The number of hidden units in the MLP. Defaults to None.
            By default, use an MLP expansion ratio of 4/3. This ratio is used by
            Tucker et al. (2022) <https://arxiv.org/abs/2204.09722> in their 3-layer
            MLP probes. We could also use a ratio of 4, imitating transformer FFNs,
            but this seems to lead to excessively large MLPs when num_layers > 2.
        init: The initialization scheme to use. Defaults to "zero".
        loss: The loss function to use. Defaults to "squared".
        num_layers: The number of layers in the MLP. Defaults to 1.
        pre_ln: Whether to include a LayerNorm module before the first linear
            layer. Defaults to False.
        supervised_weight: The weight of the supervised loss. Defaults to 0.0.
    """

    activation: Literal["gelu", "relu", "swish"] = "gelu"
    bias: bool = True
    hidden_size: Optional[int] = None
    init: Literal["default", "spherical", "zero"] = "default"
    loss: Literal["js", "squared"] = "squared"
    num_layers: int = 1
    pre_ln: bool = False
    seed: int = 42
    supervised_weight: float = 0.0


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


class Reporter(nn.Module):
    """An ELK reporter network.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    def __init__(
        self, in_features: int, cfg: ReporterConfig, device: Optional[str] = None
    ):
        super().__init__()

        hidden_size = cfg.hidden_size or 4 * in_features // 3

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

        self.init = cfg.init
        self.device = device
        self.unsupervised_loss = js_loss if cfg.loss == "js" else ccs_squared_loss
        self.supervised_weight = cfg.supervised_weight

    def reset_parameters(self):
        """Reset the parameters of the probe.

        If init is "spherical", use the spherical initialization scheme.
        If init is "default", use the default PyTorch initialization scheme for
        nn.Linear (Kaiming uniform).
        If init is "zero", initialize all parameters to zero.
        """
        if self.init == "spherical":
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

        elif self.init == "default":
            for layer in self.probe:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        elif self.init == "zero":
            for param in self.parameters():
                param.data.zero_()
        else:
            raise ValueError(f"Unknown init: {self.init}")

    # TODO: These methods will do something fancier in the future
    @classmethod
    def load(cls, path: Union[Path, str]):
        """Load a reporter from a file."""
        return torch.load(path)

    def save(self, path: Union[Path, str]):
        # TODO: Save separate JSON and PT files for the reporter.
        torch.save(self, path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw score output of the probe on `x`."""
        return self.probe(x).squeeze(-1)

    def loss(
        self,
        logit0: torch.Tensor,
        logit1: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the loss of the reporter on the contrast pair (x0, x1).

        Args:
            logit0: The raw score output of the reporter on x0.
            logit1: The raw score output of the reporter on x1.
            labels: The labels of the contrast pair. Defaults to None.

        Returns:
            loss: The loss of the reporter on the contrast pair (x0, x1).

        Raises:
            ValueError: If `supervised_weight > 0` but `labels` is None.
        """
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
        """Validate that the data's shape is valid."""
        assert len(data) == 2 and data[0].shape == data[1].shape

    def fit(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        cfg: OptimConfig = OptimConfig(),
    ) -> float:
        """Fit the probe to the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.
            labels: The labels of the contrast pair. Defaults to None.
            lr: The learning rate for Adam. Defaults to 1e-2.
            num_epochs: The number of epochs to train for. Defaults to 1000.
            num_tries: The number of times to repeat the procedure. Defaults to 10.
            optimizer: The optimizer to use. Defaults to "adam".
            verbose: Whether to print out information at each step. Defaults to False.
            weight_decay: The weight decay for Adam. Defaults to 0.01.

        Returns:
            best_loss: The best loss obtained.

        Raises:
            ValueError: If `optimizer` is not "adam" or "lbfgs".
            RuntimeError: If the best loss is not finite.
        """
        self.validate_data(contrast_pair)

        # Record the best acc, loss, and params found so far
        best_loss = torch.inf
        best_state: dict[str, torch.Tensor] = {}  # State dict of the best run
        x0, x1 = contrast_pair

        for _ in range(cfg.num_tries):
            self.reset_parameters()

            if cfg.optimizer == "lbfgs":
                loss = self.train_loop_lbfgs(x0, x1, labels, cfg)
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

    @torch.no_grad()
    def score(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> EvalResult:
        """Score the probe on the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.
            labels: The labels of the contrast pair.

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on the contrast pair (x0, x1).
        """

        self.validate_data(contrast_pair)

        logit0, logit1 = map(self, contrast_pair)
        p0, p1 = logit0.sigmoid(), logit1.sigmoid()
        pred_probs = 0.5 * (p0 + (1 - p1))

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
        cfg: OptimConfig,
    ) -> float:
        """Adam train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        loss = torch.inf
        for _ in range(cfg.num_epochs):
            optimizer.zero_grad()

            loss = self.loss(self(x0), self(x1), labels)
            loss.backward()
            optimizer.step()

        return float(loss)

    def train_loop_lbfgs(
        self, x0, x1, labels: Optional[torch.Tensor], cfg: OptimConfig
    ) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=cfg.num_epochs,
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
                regularizer += cfg.weight_decay * param.norm() ** 2 / 2

            regularized = loss + regularizer
            regularized.backward()

            return float(regularized)

        optimizer.step(closure)
        return float(loss)
