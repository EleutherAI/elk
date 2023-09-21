from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
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


def get_logprobs(
    y_logits: Tensor, ensembling: Literal["none", "full"] = "none"
) -> Tensor:
    """
    Get the class probabilities from a tensor of logits.
    Args:
        y_logits: Predicted log-odds of the positive class, tensor of shape (n, v).
    Returns:
        Tensor of logprobs: If ensemble is "none", a tensor of shape (n, v).
            If ensemble is "full", a tensor of shape (n,).
    """
    if ensembling == "full":
        y_logits = y_logits.mean(dim=1)
    return F.logsigmoid(y_logits)


def evaluate_preds(
    y_true: Tensor,
    y_logits: Tensor,
    ensembling: Literal["none", "full"] = "none",
) -> EvalResult:
    """
    Evaluate the performance of a classification model.

    Args:
        y_true: Ground truth tensor of shape (N,).
        y_logits: Predicted class tensor of shape (N, variants).

    Returns:
        dict: A dictionary containing the accuracy, AUROC, and ECE.
    """
    (n, v) = y_logits.shape
    assert y_true.shape == (n,)

    if ensembling == "full":
        y_logits = y_logits.mean(dim=1)
    else:
        y_true = repeat(y_true, "n -> n v", v=v)

    y_pred = y_logits > 0

    auroc = roc_auc_ci(y_true.long(), y_logits)
    acc = accuracy_ci(y_true, y_pred)

    pos_probs = torch.sigmoid(y_logits)

    # Calibrated accuracy
    cal_thresh = pos_probs.float().quantile(y_true.float().mean())
    cal_preds = pos_probs.gt(cal_thresh).to(torch.int)
    cal_acc = accuracy_ci(y_true, cal_preds)

    cal = CalibrationError().update(y_true.flatten(), pos_probs.flatten())
    cal_err = cal.compute()

    return EvalResult(acc, cal_acc, cal_err, auroc)
