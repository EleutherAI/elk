import pickle
from pathlib import Path
from typing import Tuple

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor

from ..utils.typing import assert_type
from .classifier import Classifier

# TODO: Create class for baseline?


def evaluate_baseline(
    lr_model: Classifier, val_x0: Tensor, val_x1: Tensor, val_labels: Tensor
) -> Tuple[float, float]:
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


def train_baseline(
    x0: Tensor,
    x1: Tensor,
    train_labels: Tensor,
    device: str,
) -> Classifier:
    # repeat_interleave makes `num_variants` copies of each label, all within a
    # single dimension of size `num_variants * 2 * n`, such that the labels align
    # with X.view(-1, X.shape[-1])
    train_labels_aug = torch.cat([train_labels, 1 - train_labels]).repeat_interleave(
        x0.shape[1]
    )

    X = torch.cat([x0, x1]).squeeze()
    d = X.shape[-1]
    lr_model = Classifier(d, device=device)
    lr_model.fit_cv(X.view(-1, d), train_labels_aug)

    return lr_model


def save_baseline(lr_dir: Path, layer: int, lr_model: Classifier):
    with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
        pickle.dump(lr_model, file)


def load_baseline(lr_dir: Path, layer: int) -> Classifier:
    with open(lr_dir / f"layer_{layer}.pt", "rb") as file:
        return pickle.load(file)
