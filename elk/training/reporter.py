from .losses import ccs_squared_loss, js_loss, prompt_var_loss
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from simple_parsing.helpers import field, Serializable
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy as bce
from typing import ClassVar, Literal, NamedTuple, Optional, Type, Union
import torch
import torch.nn as nn


class EvalResult(NamedTuple):
    """The result of evaluating a reporter on a dataset.

    The `.score()` function of a reporter returns an instance of this class,
    which contains the loss, accuracy, and AUROC.
    """

    loss: float
    acc: float
    auroc: float


@dataclass
class OptimConfig(Serializable):
    """
    Args:
        lr: The learning rate to use. Ignored when `optimizer` is `"lbfgs"`.
            Defaults to 1e-2.
        max_iter: Maximum number of steps to train for. Defaults to 1000.
        optimizer: The optimizer to use. Defaults to "adam".
        weight_decay: The weight decay or L2 penalty to use. Defaults to 0.01.
    """

    lr: float = 1e-2
    max_iter: int = 1000
    optimizer: Literal["adam", "lbfgs"] = "lbfgs"
    weight_decay: float = 0.01


@dataclass
class ReporterConfig(Serializable):
    in_features: int = field(default=0, cmd=False)
    """The number of input features. If 0, we use `in_features` in `__init__`."""
    loss: Literal["js", "squared", "prompt_var"] = "squared"
    """The loss function to use. Defaults to "squared"."""
    num_heads: int = 1
    """The number of independent predictions to output. Defaults to 1."""
    seed: int = 42
    """The random seed to use for initialization. Defaults to 42."""
    supervised_weight: float = 0.0
    """The weight of the supervised loss. Defaults to 0.0."""


CONFIG_CLS_TO_REPORTER_CLS: dict[Type[ReporterConfig], Type["Reporter"]] = {}


class Reporter(nn.Module, ABC):
    # Subclasses must set this to the configuration class they use
    config_cls: ClassVar[Type[ReporterConfig]]
    config: ReporterConfig
    in_features: int

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        CONFIG_CLS_TO_REPORTER_CLS[cls.config_cls] = cls

    @classmethod
    def instantiate(
        cls, in_features: int, cfg: ReporterConfig, device: Optional[str] = None
    ) -> "Reporter":
        """Instantiate the appropriate reporter class from a configuration."""
        reporter_cls = CONFIG_CLS_TO_REPORTER_CLS[type(cfg)]
        return reporter_cls(in_features, cfg, device=device)

    @abstractmethod
    def __init__(
        self, in_features: int, cfg: ReporterConfig, device: Optional[str] = None
    ):
        """Initialize an ELK reporter network.

        Args:
            in_features: The number of input features.
            cfg: The reporter configuration.
            device: The device to instantiate the reporter on. Defaults to `None`.
        """
        super().__init__()

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        cfg: Optional[ReporterConfig] = None,
        *,
        device: Optional[str] = None,
    ) -> "Reporter":
        """Load a reporter checkpoint from a file."""
        path = Path(path)

        # Find the YAML file
        if not cfg:
            yaml_path = path.with_suffix(".yaml")
            cfg = cls.config_cls.load_yaml(yaml_path)

        state_dict = torch.load(path, map_location=device)
        reporter = cls(cfg.in_features, cfg)
        reporter.load_state_dict(state_dict)

        return reporter

    def save(self, path: Union[Path, str], save_cfg: bool = True):
        """Save a reporter checkpoint (and optionally its config) to disk."""
        torch.save(self.state_dict(), path)

        if save_cfg:
            # Set the in_features if needed
            self.config.in_features = self.in_features

            yaml_path = Path(path).with_suffix(".yaml")
            self.config.save_yaml(yaml_path)

    @abstractmethod
    def fit(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        cfg: OptimConfig = OptimConfig(),
    ):
        """Fit the probe to the contrast pair (x0, x1).

        Args:
            contrast_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations.
            labels: The labels of the contrast pair. Defaults to None.
            cfg: The configuration for the optimizer.

        Returns:
            best_loss: The best loss obtained.

        Raises:
            RuntimeError: If the best loss is not finite.
        """

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
        loss_fn = {
            "js": js_loss,
            "squared": ccs_squared_loss,
            "prompt_var": prompt_var_loss,
        }[self.config.loss]
        loss = loss_fn(logit0, logit1)

        # If labels are provided, use them to compute a supervised loss
        if labels is not None:
            num_labels = len(labels)
            assert num_labels <= len(logit0), "Too many labels provided"
            p0 = logit0[:num_labels].sigmoid()
            p1 = logit1[:num_labels].sigmoid()

            alpha = self.config.supervised_weight
            preds = p0.add(1 - p1).mul(0.5).squeeze(-1)
            bce_loss = bce(preds, labels.type_as(preds))
            loss = alpha * bce_loss + (1 - alpha) * loss

        elif self.config.supervised_weight > 0:
            raise ValueError(
                "Supervised weight > 0 but no labels provided to compute loss"
            )

        return loss

    def validate_data(self, data: tuple[torch.Tensor, torch.Tensor]):
        """Validate that the data's shape is valid."""
        assert len(data) == 2 and data[0].shape == data[1].shape

    @torch.no_grad()
    def score(
        self,
        contrast_pair: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> list[EvalResult]:
        """Score the reporter on the contrast pair (x0, x1).

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

        # makes `num_variants` copies of each label, all within a single
        # dimension of size `num_variants * n`, such that the labels align
        # with pred_probs.flatten()
        labels = labels.repeat_interleave(pred_probs.shape[1])
        raw_preds = pred_probs.gt(0.5).squeeze(1).to(torch.int)

        results = []
        for pred_prob in pred_probs.unbind(-1):
            # roc_auc_score only takes flattened input
            auroc = float(roc_auc_score(labels.cpu(), pred_prob.cpu().flatten()))
            raw_acc = raw_preds.flatten().eq(labels).float().mean()

            result = EvalResult(
                loss=self.loss(logit0, logit1).item(),
                acc=torch.max(raw_acc, 1 - raw_acc).item(),
                auroc=max(auroc, 1 - auroc),
            )
            results.append(result)

        return results

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
        for _ in range(cfg.max_iter):
            optimizer.zero_grad()

            loss = self.loss(self(x0), self(x1), labels)
            loss.backward()
            optimizer.step()

        return float(loss)

    def fit_once_lbfgs(
        self, x0, x1, labels: Optional[torch.Tensor], cfg: OptimConfig
    ) -> float:
        """LBFGS train loop, returning the final loss. Modifies params in-place."""

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=cfg.max_iter,
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
