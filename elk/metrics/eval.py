from dataclasses import asdict, dataclass
from typing import Literal

import torch
from einops import repeat
from torch import Tensor

from .accuracy import AccuracyResult, accuracy_ci
from .calibration import CalibrationError, CalibrationEstimate
from .roc_auc import RocAucResult, roc_auc_ci


@dataclass(frozen=True)
class EvalResult:
    """The result of evaluating a classifier."""

    accuracy: AccuracyResult
    """Top 1 accuracy, implemented for both binary and multi-class classification."""
    cal_accuracy: AccuracyResult | None
    """Calibrated accuracy, only implemented for binary classification."""
    calibration: CalibrationEstimate | None
    """Expected calibration error, only implemented for binary classification."""
    roc_auc: RocAucResult
    """Area under the ROC curve. For multi-class classification, each class is treated
    as a one-vs-rest binary classification problem."""

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert the result to a dictionary."""
        acc_dict = {f"{prefix}acc_{k}": v for k, v in asdict(self.accuracy).items()}
        cal_acc_dict = (
            {f"{prefix}cal_acc_{k}": v for k, v in asdict(self.cal_accuracy).items()}
            if self.cal_accuracy is not None
            else {}
        )
        cal_dict = (
            {f"{prefix}ece": self.calibration.ece}
            if self.calibration is not None
            else {}
        )
        auroc_dict = {f"{prefix}auroc_{k}": v for k, v in asdict(self.roc_auc).items()}
        return {**auroc_dict, **cal_acc_dict, **acc_dict, **cal_dict}


def evaluate_preds(
    y_true: Tensor,
    y_logits: Tensor,
    ensembling: Literal["none", "partial", "full"] = "none",
) -> EvalResult:
    """
    Evaluate the performance of a classification model.

    Args:
        y_true: Ground truth tensor of shape (N,).
        y_logits: Predicted class tensor of shape (N, variants, n_classes).

    Returns:
        dict: A dictionary containing the accuracy, AUROC, and ECE.
    """
    (n, v, c) = y_logits.shape
    assert y_true.shape == (n,)

    if ensembling == "full":
        y_logits = y_logits.mean(dim=1)
    else:
        y_true = repeat(y_true, "n -> n v", v=v)

    y_pred = y_logits.argmax(dim=-1)
    if ensembling == "none":
        auroc = roc_auc_ci(to_one_hot(y_true, c).long().flatten(1), y_logits.flatten(1))
    elif ensembling in ("partial", "full"):
        # Pool together the negative and positive class logits
        if c == 2:
            auroc = roc_auc_ci(y_true, y_logits[..., 1] - y_logits[..., 0])
        else:
            auroc = roc_auc_ci(to_one_hot(y_true, c).long(), y_logits)
    else:
        raise ValueError(f"Unknown mode: {ensembling}")

    acc = accuracy_ci(y_true, y_pred)
    cal_acc = None
    cal_err = None

    if c == 2:
        pos_probs = torch.sigmoid(y_logits[..., 1] - y_logits[..., 0])

        # Calibrated accuracy
        cal_thresh = pos_probs.float().quantile(y_true.float().mean())
        cal_preds = pos_probs.gt(cal_thresh).to(torch.int)
        cal_acc = accuracy_ci(y_true, cal_preds)

        cal = CalibrationError().update(y_true.flatten(), pos_probs.flatten())
        cal_err = cal.compute()

    return EvalResult(acc, cal_acc, cal_err, auroc)


def to_one_hot(labels: Tensor, n_classes: int) -> Tensor:
    """
    Convert a tensor of class labels to a one-hot representation.

    Args:
        labels (Tensor): A tensor of class labels of shape (N,).
        n_classes (int): The total number of unique classes.

    Returns:
        Tensor: A one-hot representation tensor of shape (N, n_classes).
    """
    one_hot_labels = labels.new_zeros(*labels.shape, n_classes)
    return one_hot_labels.scatter_(-1, labels.unsqueeze(-1).long(), 1)
