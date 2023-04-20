import torch
from einops import rearrange, repeat

from ..metrics import to_one_hot
from .classifier import Classifier


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
