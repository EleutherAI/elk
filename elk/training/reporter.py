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
from ..metrics import mean_auc, to_one_hot
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
        *hiddens: Tensor,
        labels: Optional[Tensor] = None,
    ) -> float:
        ...

    @abstractmethod
    def predict(self, *hiddens: Tensor) -> Tensor:
        """Return pooled logits for the contrast set `hiddens`."""

    @abstractmethod
    def predict_prob(self, *hiddens: Tensor) -> Tensor:
        """Like `predict` but returns normalized probabilities, not logits."""

    @torch.no_grad()
    def score(self, labels: Tensor, hiddens: Tensor) -> EvalResult:
        """Score the probe on the contrast set `hiddens`.

        Args:
        labels: The labels of the contrast pair.
        hiddens: The hidden representations of the contrast set.

        Returns:
            an instance of EvalResult containing the loss, accuracy, calibrated
                accuracy, and AUROC of the probe on `hiddens`.
        """
        pred_probs = self.predict_prob(hiddens)
        (_, v, c) = pred_probs.shape

        # makes `num_variants` copies of each label
        Y = repeat(labels, "n -> (n v)", v=v).float()
        to_one_hot(Y, n_classes=c).long().flatten()

        if c == 2:
            cal_err = CalibrationError().update(Y.cpu(), pred_probs.cpu()).compute().ece
            # Calibrated accuracy
            cal_thresh = pred_probs.float().quantile(labels.float().mean())
            cal_preds = pred_probs.gt(cal_thresh).squeeze(1).to(torch.int)
            cal_acc = cal_preds.flatten().eq(Y).float().mean().item()

            raw_preds = pred_probs.gt(0.5).squeeze(1).to(torch.int)
        else:
            # TODO: Implement calibration error for k > 2?
            cal_acc = 0.0
            cal_err = 0.0

            raw_preds = pred_probs.argmax(dim=-1)

        # roc_auc_score only takes flattened input
        auroc = mean_auc(
            Y.cpu(), rearrange(pred_probs.cpu(), "n v ... -> (n v) ..."), curve="roc"
        )
        raw_acc = raw_preds.flatten().eq(Y).float().mean()

        return EvalResult(
            acc=raw_acc.item(),
            cal_acc=cal_acc,
            auroc=float(auroc),
            ece=cal_err,
        )
