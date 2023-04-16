import torch
from einops import rearrange, repeat
from sklearn.metrics import roc_auc_score
from torch import Tensor

from ..metrics import accuracy, to_one_hot
from ..utils import assert_type
from .classifier import Classifier


def evaluate_supervised(
    lr_model: Classifier, val_h: Tensor, val_labels: Tensor
) -> tuple[float, float]:
    (n, v, k, d) = val_h.shape

    with torch.no_grad():
        logits = rearrange(lr_model(val_h).cpu().squeeze(), "n v k -> (n v) k")
        raw_preds = to_one_hot(logits.argmax(dim=-1), k).long()

    labels = repeat(val_labels, "n -> (n v)", v=v)
    labels = to_one_hot(labels, k).flatten()

    lr_acc = accuracy(labels, raw_preds.flatten())
    lr_auroc = roc_auc_score(labels.cpu(), logits.cpu().flatten())

    return assert_type(float, lr_auroc), assert_type(float, lr_acc)


def train_supervised(data: dict[str, tuple], device: str, cv: bool) -> Classifier:
    Xs, train_labels = [], []

    for train_h, labels, _ in data.values():
        (_, v, k, _) = train_h.shape
        train_h = rearrange(train_h, "n v k d -> (n v k) d")

        labels = repeat(labels, "n -> (n v)", v=v)
        labels = to_one_hot(labels, k).flatten()

        Xs.append(train_h)
        train_labels.append(labels)

    X, train_labels = torch.cat(Xs), torch.cat(train_labels)
    lr_model = Classifier(X.shape[-1], device=device)
    if cv:
        lr_model.fit_cv(X, train_labels)
    else:
        lr_model.fit(X, train_labels)

    return lr_model
