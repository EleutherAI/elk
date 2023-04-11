from functools import partial
from typing import Literal

from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor


def to_one_hot(labels: Tensor, n_classes: int) -> Tensor:
    """
    Convert a tensor of class labels to a one-hot representation.

    Args:
        labels (Tensor): A tensor of class labels of shape (N,).
        n_classes (int): The total number of unique classes.

    Returns:
        Tensor: A one-hot representation tensor of shape (N, n_classes).
    """
    one_hot_labels = labels.new_zeros(labels.size(0), n_classes)
    return one_hot_labels.scatter_(1, labels.unsqueeze(1).long(), 1)


def accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    """
    Compute the accuracy of a classification model.

    Args:
        y_true: Ground truth tensor of shape (N,).
        y_pred: Predicted class tensor of shape (N,) or (N, n_classes).

    Returns:
        float: Accuracy of the model.
    """
    # Check if binary or multi-class classification
    if len(y_pred.shape) == 1:
        hard_preds = y_pred > 0.5
    else:
        hard_preds = y_pred.argmax(-1)

    return hard_preds.eq(y_true).float().mean().item()


def mean_auc(y_true: Tensor, y_scores: Tensor, curve: Literal["roc", "pr"]) -> float:
    """
    Compute the mean area under the receiver operating curve (AUROC) or
    precision-recall curve (average precision or mAP) for binary or multi-class
    classification problems.

    Args:
        y_true: Ground truth tensor of shape (N,) or (N, n_classes).
        y_scores: Predicted probability tensor of shape (N,) for binary
            or (N, n_classes) for multi-class.
        curve: Type of curve to compute the mean AUC. Either 'pr' for
            precision-recall curve or 'roc' for receiver operating
            characteristic curve. Defaults to 'pr'.

    Returns:
        float: Either mean AUROC or mean average precision (mAP).
    """
    score_fn = {
        "pr": average_precision_score,
        "roc": partial(roc_auc_score, multi_class="ovo"),
    }.get(curve, None)

    if score_fn is None:
        raise ValueError("Invalid curve type. Supported values are 'pr' and 'roc'.")

    if len(y_scores.shape) == 1 or y_scores.shape[1] == 1:
        return float(score_fn(y_true, y_scores.squeeze(1)))
    else:
        n_classes = y_scores.shape[1]
        y_true_one_hot = to_one_hot(y_true, n_classes)

        return score_fn(y_true_one_hot, y_scores)
        # return np.array([
        #     score_fn(y_true_one_hot[:, i], y_scores[:, i])
        #     for i in range(n_classes)
        # ]).mean()
