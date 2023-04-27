"""An ELK reporter network."""

from ..calibration import CalibrationError
from .classifier import Classifier
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from simple_parsing.helpers import Serializable
from sklearn.metrics import roc_auc_score
from torch import Tensor
from typing import Literal, NamedTuple, Optional, Union
import torch
import torch.nn as nn


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
    neg_mean: Tensor
    pos_mean: Tensor

    def __init__(
        self,
        in_features: int,
        cfg: ReporterConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.config = cfg
        self.register_buffer("n", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer(
            "neg_mean", torch.zeros(in_features, device=device, dtype=dtype)
        )
        self.register_buffer(
            "pos_mean", torch.zeros(in_features, device=device, dtype=dtype)
        )

    @classmethod
    def check_separability(
        cls,
        train_pair: tuple[Tensor, Tensor],
        val_pair: tuple[Tensor, Tensor],
    ) -> float:
        """Measure how linearly separable the pseudo-labels are for a contrast pair.

        Args:
            train_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations. Used for training the classifier.
            val_pair: A tuple of tensors, (x0, x1), where x0 and x1 are the
                contrastive representations. Used for evaluating the classifier.

        Returns:
            The AUROC of a linear classifier fit on the pseudo-labels.
        """
        x0, x1 = train_pair
        val_x0, val_x1 = val_pair

        pseudo_clf = Classifier(x0.shape[-1], device=x0.device)  # type: ignore
        pseudo_train_labels = torch.cat(
            [
                x0.new_zeros(x0.shape[0]),
                x0.new_ones(x0.shape[0]),
            ]
        ).repeat_interleave(
            x0.shape[1]
        )  # make num_variants copies of each pseudo-label
        pseudo_val_labels = torch.cat(
            [
                val_x0.new_zeros(val_x0.shape[0]),
                val_x0.new_ones(val_x0.shape[0]),
            ]
        ).repeat_interleave(val_x0.shape[1])

        pseudo_clf.fit(
            # b v d -> (b v) d
            torch.cat([x0, x1]).flatten(0, 1),
            pseudo_train_labels,
        )
        with torch.no_grad():
            pseudo_preds = pseudo_clf(
                # b v d -> (b v) d
                torch.cat([val_x0, val_x1]).flatten(0, 1)
            )
            return float(roc_auc_score(pseudo_val_labels.cpu(), pseudo_preds.cpu()))

    def reset_parameters(self):
        """Reset the parameters of the probe."""

    @torch.no_grad()
    def update(self, x_pos: Tensor, x_neg: Tensor) -> None:
        """Update the running mean of the positive and negative examples."""

        x_pos, x_neg = x_pos.flatten(0, -2), x_neg.flatten(0, -2)
        self.n += x_pos.shape[0]

        # Update the running means
        self.neg_mean += (x_neg.sum(dim=0) - self.neg_mean) / self.n
        self.pos_mean += (x_pos.sum(dim=0) - self.pos_mean) / self.n

    # TODO: These methods will do something fancier in the future
    @classmethod
    def load(cls, path: Union[Path, str]):
        """Load a reporter from a file."""
        return torch.load(path)

    def save(self, path: Union[Path, str]):
        # TODO: Save separate JSON and PT files for the reporter.
        torch.save(self, path)

    @abstractmethod
    def fit(
        self,
        x_pos: Tensor,
        x_neg: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        ...

    @abstractmethod
    def predict(self, x_pos: Tensor, x_neg: Tensor) -> Tensor:
        """Pool the probe output on the contrast pair (x_pos, x_neg)."""

    @torch.no_grad()
    def score(self, labels: Tensor, x_pos: Tensor, x_neg: Tensor) -> EvalResult:
        """Score the probe on the contrast pair (x_pos, x1).

        Args:
            x_pos: The positive examples.
            x_neg: The negative examples.
            labels: The labels of the contrast pair.

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on the contrast pair (x0, x1).
        """

        pred_probs = self.predict(x_pos, x_neg)

        # makes `num_variants` copies of each label, all within a single
        # dimension of size `num_variants * n`, such that the labels align
        # with pred_probs.flatten()
        broadcast_labels = labels.repeat_interleave(pred_probs.shape[1]).float()
        cal_err = (
            CalibrationError()
            .update(broadcast_labels.cpu(), pred_probs.cpu())
            .compute()
        )

        # Calibrated accuracy
        cal_thresh = pred_probs.float().quantile(labels.float().mean())
        cal_preds = pred_probs.gt(cal_thresh).squeeze(1).to(torch.int)
        raw_preds = pred_probs.gt(0.5).squeeze(1).to(torch.int)

        # roc_auc_score only takes flattened input
        auroc = float(roc_auc_score(broadcast_labels.cpu(), pred_probs.cpu().flatten()))
        cal_acc = cal_preds.flatten().eq(broadcast_labels).float().mean()
        raw_acc = raw_preds.flatten().eq(broadcast_labels).float().mean()

        return EvalResult(
            acc=torch.max(raw_acc, 1 - raw_acc).item(),
            cal_acc=torch.max(cal_acc, 1 - cal_acc).item(),
            auroc=max(auroc, 1 - auroc),
            ece=cal_err.ece,
        ), cal_preds, raw_preds
