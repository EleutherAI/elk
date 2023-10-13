from dataclasses import asdict, dataclass

import torch
from einops import repeat
from torch import Tensor

from ..utils.types import PromptEnsembling
from .accuracy import AccuracyResult, accuracy_ci
from .calibration import CalibrationError, CalibrationEstimate
from .roc_auc import RocAucResult, roc_auc_ci


@dataclass
class LayerOutput:
    val_gt: Tensor
    val_credences: Tensor
    meta: dict


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
    cal_thresh: float | None
    """The threshold used to compute the calibrated accuracy."""

    def to_dict(self, prefix: str = "") -> dict[str, float | None]:
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
        return {
            **auroc_dict,
            **cal_acc_dict,
            **acc_dict,
            **cal_dict,
            f"{prefix}cal_thresh": self.cal_thresh,
        }


def calc_auroc(
    y_logits: Tensor,
    y_true: Tensor,
    ensembling: PromptEnsembling,
    num_classes: int,
) -> RocAucResult:
    """
    Calculate the AUROC

    Args:
        y_true: Ground truth tensor of shape (n,).
        y_logits: Predicted class tensor of shape (n, num_variants, num_classes).
        prompt_ensembling: The prompt_ensembling mode.
        num_classes: The number of classes.

    Returns:
        RocAucResult: A dictionary containing the AUROC and confidence interval.
    """
    if ensembling == PromptEnsembling.NONE:
        auroc = roc_auc_ci(
            to_one_hot(y_true, num_classes).long().flatten(1), y_logits.flatten(1)
        )
    elif ensembling in (PromptEnsembling.PARTIAL, PromptEnsembling.FULL):
        # Pool together the negative and positive class logits
        if num_classes == 2:
            auroc = roc_auc_ci(y_true, y_logits[..., 1] - y_logits[..., 0])
        else:
            auroc = roc_auc_ci(to_one_hot(y_true, num_classes).long(), y_logits)
    else:
        raise ValueError(f"Unknown mode: {ensembling}")

    return auroc


def calc_calibrated_accuracies(y_true, pos_probs) -> AccuracyResult:
    """
    Calculate the calibrated accuracies

    Args:
        y_true: Ground truth tensor of shape (n,).
        pos_probs: Predicted class tensor of shape (n, num_variants, num_classes).

    Returns:
        AccuracyResult: A dictionary containing the accuracy and confidence interval.
    """

    cal_thresh = pos_probs.float().quantile(y_true.float().mean()).item()
    cal_preds = pos_probs.gt(cal_thresh).to(torch.int)
    cal_acc = accuracy_ci(y_true, cal_preds, cal_thresh)
    return cal_acc


def calc_calibrated_errors(y_true, pos_probs) -> CalibrationEstimate:
    """
    Calculate the expected calibration error.

    Args:
        y_true: Ground truth tensor of shape (n,).
        y_logits: Predicted class tensor of shape (n, num_variants, num_classes).

    Returns:
        CalibrationEstimate:
    """

    cal = CalibrationError().update(y_true.flatten(), pos_probs.flatten())
    cal_err = cal.compute()
    return cal_err


def calc_accuracies(y_logits, y_true) -> AccuracyResult:
    """
    Calculate the accuracy

    Args:
        y_true: Ground truth tensor of shape (n,).
        y_logits: Predicted class tensor of shape (n, num_variants, num_classes).

    Returns:
        AccuracyResult: A dictionary containing the accuracy and confidence interval.
    """
    y_pred = y_logits.argmax(dim=-1)
    return accuracy_ci(y_true, y_pred)


def evaluate_preds(
    y_true: Tensor,
    y_logits: Tensor,
    prompt_ensembling: PromptEnsembling = PromptEnsembling.NONE,
) -> EvalResult:
    """
    Evaluate the performance of a classification model.

    Args:
        y_true: Ground truth tensor of shape (n,).
        y_logits: Predicted class tensor of shape (n, num_variants, num_classes).
        prompt_ensembling: The prompt_ensembling mode.

    Returns:
        dict: A dictionary containing the accuracy, AUROC, and ECE.
    """
    y_logits, y_true, num_classes = prepare(y_logits, y_true, prompt_ensembling)
    return calc_eval_results(y_true, y_logits, prompt_ensembling, num_classes)


def prepare(y_logits: Tensor, y_true: Tensor, prompt_ensembling: PromptEnsembling):
    """
    Prepare the logits and ground truth for evaluation
    """
    (n, num_variants, num_classes) = y_logits.shape
    assert y_true.shape == (n,), f"y_true.shape: {y_true.shape} is not equal to n: {n}"

    if prompt_ensembling == PromptEnsembling.FULL:
        y_logits = y_logits.mean(dim=1)
    else:
        y_true = repeat(y_true, "n -> n v", v=num_variants)

    return y_logits, y_true, num_classes


def calc_eval_results(
    y_true: Tensor,
    y_logits: Tensor,
    prompt_ensembling: PromptEnsembling,
    num_classes: int,
) -> EvalResult:
    """
    Calculate the evaluation results

    Args:
        y_true: Ground truth tensor of shape (n,).
        y_logits: Predicted class tensor of shape (n, num_variants, num_classes).
        prompt_ensembling: The prompt_ensembling mode.

    Returns:
        EvalResult: The result of evaluating a classifier containing the accuracy,
        calibrated accuracies, calibrated errors, and AUROC.
    """
    acc = calc_accuracies(y_logits=y_logits, y_true=y_true)

    pos_probs = torch.sigmoid(y_logits[..., 1] - y_logits[..., 0])
    cal_acc, cal_thresh = (
        calc_calibrated_accuracies(y_true=y_true, pos_probs=pos_probs)
        if num_classes == 2
        else None,
        None,
    )
    cal_err = (
        calc_calibrated_errors(y_true=y_true, pos_probs=pos_probs)
        if num_classes == 2
        else None
    )

    auroc = calc_auroc(
        y_logits=y_logits,
        y_true=y_true,
        ensembling=prompt_ensembling,
        num_classes=num_classes,
    )

    return EvalResult(acc, cal_acc, cal_err, auroc, cal_thresh)


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


def layer_ensembling(
    layer_outputs: list[LayerOutput], prompt_ensembling: PromptEnsembling
) -> EvalResult:
    """
    Return EvalResult after prompt_ensembling
    the probe output of the middle to last layers

    Args:
        layer_outputs: A list of LayerOutput containing the ground truth and
        predicted class tensor of shape (n, num_variants, num_classes).
        prompt_ensembling: The prompt_ensembling mode.

    Returns:
        EvalResult: The result of evaluating a classifier containing the accuracy,
        calibrated accuracies, calibrated errors, and AUROC.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_logits_collection = []

    num_classes = 2
    y_true = layer_outputs[0].val_gt.to(device)

    for layer_output in layer_outputs:
        # all y_trues are identical, so just get the first
        y_logits = layer_output.val_credences.to(device)
        y_logits, y_true, num_classes = prepare(
            y_logits=y_logits,
            y_true=layer_outputs[0].val_gt.to(device),
            prompt_ensembling=prompt_ensembling,
        )
        y_logits_collection.append(y_logits)

    # get logits and ground_truth from middle to last layer
    middle_index = len(layer_outputs) // 2
    y_logits_stacked = torch.stack(y_logits_collection[middle_index:])
    # layer prompt_ensembling of the stacked logits
    y_logits_stacked_mean = torch.mean(y_logits_stacked, dim=0)

    return calc_eval_results(
        y_true=y_true,
        y_logits=y_logits_stacked_mean,
        prompt_ensembling=prompt_ensembling,
        num_classes=num_classes,
    )
