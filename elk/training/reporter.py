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
from ..metrics import to_one_hot
from .classifier import Classifier


class EvalResult(NamedTuple):
    """The result of evaluating a reporter on a dataset.

    The `.score()` function of a reporter returns an instance of this class,
    which contains the loss, accuracy, calibrated accuracy, and AUROC.
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
    """An ELK reporter network.

    Args:
        in_features: The number of input features.
        cfg: The reporter configuration.
    """

    n: Tensor
    class_means: Tensor

    def __init__(
        self,
        cfg: ReporterConfig,
        in_features: int,
        num_classes: int = 2,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.config = cfg
        self.register_buffer("n", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer(
            "class_means",
            torch.zeros(num_classes, in_features, device=device, dtype=dtype),
        )

    @classmethod
    def check_separability(
        cls,
        train_hiddens: Tensor,
        val_hiddens: Tensor,
    ) -> float:
        """Measure how linearly separable the pseudo-labels are for a contrast pair.

        Args:
            train_hiddens: Contrast set of shape [n, v, k, d]. Used for training the
                classifier.
            val_hiddens: Contrast set of shape [n, v, k, d]. Used for evaluating the
                classifier.

        Returns:
            The AUROC of a linear classifier fit on the pseudo-labels.
        """
        (n_train, v, k, d) = train_hiddens.shape
        (n_val, _, k_val, d_val) = val_hiddens.shape
        assert d == d_val, "Must have the same number of features in each split"
        assert k == k_val == 2, "Must be a binary contrast set"

        pseudo_clf = Classifier(d, device=train_hiddens.device)
        pseudo_train_labels = torch.cat(
            [
                train_hiddens.new_zeros(n_train),
                train_hiddens.new_ones(n_train),
            ]
        ).repeat_interleave(
            v
        )  # make num_variants copies of each pseudo-label

        pseudo_val_labels = torch.cat(
            [
                val_hiddens.new_zeros(n_val),
                val_hiddens.new_ones(n_val),
            ]
        ).repeat_interleave(v)

        pseudo_clf.fit(
            rearrange(train_hiddens, "n v k d -> (k n v) d"),
            pseudo_train_labels,
        )
        with torch.no_grad():
            pseudo_preds = pseudo_clf(
                rearrange(val_hiddens, "n v k d -> (k n v) d"),
            )
            return float(roc_auc_score(pseudo_val_labels.cpu(), pseudo_preds.cpu()))

    def reset_parameters(self):
        """Reset the parameters of the probe."""

    @torch.no_grad()
    def update(self, *hiddens: Tensor) -> None:
        """Update the running mean of the positive and negative examples."""

        assert len(hiddens) > 1, "Must provide at least two hidden representations"

        # Flatten the hidden representations
        hiddens = tuple(h.flatten(0, -2) for h in hiddens)
        self.n += hiddens[0].shape[0]

        # Update the running means
        for i, h in enumerate(hiddens):
            self.class_means[i] += (h.sum(dim=0) - self.class_means[i]) / self.n

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
    def score(self, labels: Tensor, hiddens: Tensor) -> EvalResult:
        """Score the probe on the contrast set `hiddens`.

        Args:
        labels: The labels of the contrast pair.
        hiddens: Contrast set of shape [n, v, k, d].

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on `hiddens`.
        """
        logits = self(hiddens)
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
        auroc = roc_auc_score(
            to_one_hot(Y, c).long().flatten().cpu(), logits.cpu().flatten()
        )
        raw_acc = raw_preds.flatten().eq(Y).float().mean()

        return EvalResult(
            acc=raw_acc.item(),
            cal_acc=cal_acc,
            auroc=float(auroc),
            ece=cal_err,
        )
