import pickle
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch import Tensor

from ..metrics import accuracy, mean_auc, to_one_hot
from ..utils.typing import assert_type
from .classifier import Classifier

# TODO: Create class for baseline?


def evaluate_baseline(
    lr_model: Classifier, hiddens: Tensor, labels: Tensor
) -> tuple[float, float]:
    # n = batch, v = variants, c = classes, d = hidden dim
    (_, v, c, _) = hiddens.shape

    Y = repeat(labels, "n -> (n v)", v=v)
    Y_one_hot = to_one_hot(Y, n_classes=c).long().flatten()
    X = rearrange(hiddens, "n v c d -> (n v c) d")
    with torch.no_grad():
        lr_preds = lr_model(X)

    # Top-1 accuracy
    lr_acc = accuracy(
        Y.cpu(), rearrange(lr_preds.squeeze(-1), "(n v c) -> (n v) c", v=v, c=c).cpu()
    )
    lr_auroc = mean_auc(Y_one_hot.cpu(), lr_preds.cpu(), curve="roc")

    return assert_type(float, lr_auroc), assert_type(float, lr_acc)


def train_baseline(hiddens: Tensor, labels: Tensor) -> Classifier:
    # n = batch, v = variants, c = classes, d = hidden dim
    (_, v, c, d) = hiddens.shape

    Y = repeat(labels, "n -> (n v)", v=v)
    Y = to_one_hot(Y, n_classes=c).long().flatten()
    X = rearrange(hiddens, "n v c d -> (n v c) d")

    lr_model = Classifier(d, device=X.device)
    lr_model.fit(X, Y)

    return lr_model


def save_baseline(lr_dir: Path, layer: int, lr_model: Classifier):
    with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
        pickle.dump(lr_model, file)


def load_baseline(lr_dir: Path, layer: int) -> Classifier:
    with open(lr_dir / f"layer_{layer}.pt", "rb") as file:
        return pickle.load(file)
