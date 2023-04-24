import torch
from einops import rearrange, repeat
from torch import Tensor

from ..metrics import RocAucResult, accuracy, roc_auc_ci, to_one_hot
from ..utils import assert_type
from .classifier import Classifier


def evaluate_supervised(
    lr_model: Classifier, val_h: Tensor, val_labels: Tensor
) -> tuple[RocAucResult, float]:
    if len(val_h.shape) == 4:
        # hiddens in a contrast set
        (_, v, c, _) = val_h.shape

        with torch.no_grad():
            logits = rearrange(lr_model(val_h).squeeze(), "n v c -> (n v) c")
            raw_preds = to_one_hot(logits.argmax(dim=-1), c).long()

        labels = repeat(val_labels, "n -> (n v)", v=v)
        labels = to_one_hot(labels, c).flatten()
    elif len(val_h.shape) == 2:
        # raw hiddens
        (n, d) = val_h.shape

        with torch.no_grad():
            logits = lr_model(val_h).cpu().flatten()
            probs = logits.sigmoid()
            raw_preds = probs.gt(0.5).long()

        labels = val_labels
    else:
        raise ValueError(f"Invalid val_h shape: {val_h.shape}")

    lr_acc = accuracy(labels, raw_preds.flatten())
    lr_auroc = (
        RocAucResult(-1.0, -1.0, -1.0)
        if len(labels.unique()) == 1
        else roc_auc_ci(labels.cpu(), logits.cpu().flatten())
    )

    return lr_auroc, assert_type(float, lr_acc)


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
