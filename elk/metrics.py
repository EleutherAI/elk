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

    return hard_preds.cpu().eq(y_true.cpu()).float().mean().item()
