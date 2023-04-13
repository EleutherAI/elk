import torch
from einops import rearrange, repeat
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor

from ..utils import assert_type
from .classifier import Classifier


def evaluate_supervised(
    lr_model: Classifier, val_x0: Tensor, val_x1: Tensor, val_labels: Tensor
) -> tuple[float, float]:
    X = torch.cat([val_x0, val_x1])
    d = X.shape[-1]
    X_val = X.view(-1, d)
    with torch.no_grad():
        lr_preds = lr_model(X_val).sigmoid().cpu()

    val_labels_aug = (
        torch.cat([val_labels, 1 - val_labels]).repeat_interleave(val_x0.shape[1])
    ).cpu()

    lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
    lr_auroc = roc_auc_score(val_labels_aug, lr_preds)

    return assert_type(float, lr_auroc), assert_type(float, lr_acc)


def train_supervised(data: dict[str, tuple], device: str) -> Classifier:
    Xs, train_labels = [], []

    for x0, x1, labels, _ in data.values():
        (_, v, _) = x0.shape
        x0 = rearrange(x0, "n v d -> (n v) d")
        x1 = rearrange(x1, "n v d -> (n v) d")

        labels = repeat(labels, "n -> (n v)", v=v)
        labels = torch.cat([labels, 1 - labels])

        Xs.append(torch.cat([x0, x1]).squeeze())
        train_labels.append(labels)

    X, train_labels = torch.cat(Xs), torch.cat(train_labels)
    lr_model = Classifier(X.shape[-1], device=device)
    lr_model.fit_cv(X, train_labels)

    return lr_model
