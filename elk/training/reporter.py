"""An ELK reporter network."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from simple_parsing.helpers import Serializable
from sklearn.metrics import roc_auc_score
from torch import Tensor

from ..calibration import CalibrationError
from ..metrics import accuracy, to_one_hot


class EvalResult(NamedTuple):
    """The result of evaluating a reporter on a dataset.

    The `.score()` and `.score_contrast_set` functions of a reporter
    returns an instance of this class, which contains the loss,
    accuracy, calibrated accuracy, and AUROC.
    """

    acc: float
    cal_acc: float
    auroc: float
    ece: float


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

    @torch.no_grad()
    def score_contrast_set(self, labels: Tensor, contrast_set: Tensor) -> EvalResult:
        """Score the probe on a contrast set `contrast_set`.

        Args:
        labels: The labels of the contrast pair.
        contrast_set: Contrast set of shape [n, v, c, d]
            if c = 2, then 0 is the negative choice and 1 is positive.

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on `contrast_set`.
        """
        logits = self(contrast_set)
        (_, v, c) = logits.shape

        # makes `num_variants` copies of each label
        logits = rearrange(logits, "n v c -> (n v) c")
        Y = repeat(labels, "n -> (n v)", v=v).float()

        if c == 2:
            pos_probs = logits[..., 1].flatten().sigmoid()
            cal_err = CalibrationError().update(Y.cpu(), pos_probs.cpu()).compute().ece

            # Calibrated accuracy
            cal_thresh = pos_probs.float().quantile(labels.float().mean())
            cal_preds = pos_probs.gt(cal_thresh).to(torch.int)
            cal_acc = cal_preds.flatten().eq(Y).float().mean().item()
        else:
            # TODO: Implement calibration error for k > 2?
            cal_acc = 0.0
            cal_err = 0.0

        raw_preds = to_one_hot(logits.argmax(dim=-1), c).long()
        Y = to_one_hot(Y, c).long().flatten()

        auroc = roc_auc_score(Y.cpu(), logits.cpu().flatten())
        # TODO: doesn't this acc measure inflate the accuracy because
        # we enforce mutual exclusivity?
        raw_acc = accuracy(Y, raw_preds.flatten())

        return EvalResult(
            acc=float(raw_acc),
            cal_acc=cal_acc,
            auroc=float(auroc),
            ece=cal_err,
        )

    @torch.no_grad()
    def score(self, labels: Tensor, hiddens: Tensor) -> EvalResult:
        """Score the probe on `hiddens`.

        Args:
            labels: The labels of the contrast pair.
            hiddens: Hiddens of shape [n, d].

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on `hiddens`.
        """
        assert hiddens.shape[0] == labels.shape[0]
        assert len(hiddens.shape) == 2
        logits = self(hiddens)

        probs = logits.flatten().sigmoid()
        cal_err = CalibrationError().update(labels.cpu(), probs.cpu()).compute().ece

        # Calibrated accuracy
        cal_thresh = probs.float().quantile(labels.float().mean())
        cal_preds = probs.gt(cal_thresh).to(torch.int)
        cal_acc = cal_preds.flatten().eq(labels).float().mean().item()

        preds = probs.gt(0.5).to(torch.int)
        acc = preds.flatten().eq(labels).float().mean().item()

        auroc = roc_auc_score(labels.cpu(), logits.cpu().flatten())

        return EvalResult(
            acc=float(acc),
            cal_acc=cal_acc,
            auroc=float(auroc),
            ece=cal_err,
        )
