import torch
from concept_erasure import LeaceFitter
from einops import rearrange, repeat

from ..run import LayerData
from .classifier import Classifier


def train_supervised(
    data: dict[str, LayerData],
    device: str,
    mode: str,
    erase_paraphrases: bool = False,
    max_inlp_iter: int | None = None,
) -> list[Classifier]:
    assert not (
        erase_paraphrases and len(data) > 1
    ), "Erasing paraphrases is only supported for single dataset."
    Xs, train_labels = [], []

    leace = None

    for train_data in data.values():
        (n, v, d) = train_data.hiddens.shape
        train_h = rearrange(train_data.hiddens, "n v d -> (n v) d")

        if erase_paraphrases:
            if leace is None:
                leace = LeaceFitter(
                    d,
                    v,
                    device=device,
                    dtype=train_h.dtype,
                )
            # indicators = [0, 1, ..., v-1, 0, 1, ..., v-1, ...] to one-hot
            indicators = torch.eye(v, device=device, dtype=train_h.dtype).repeat(
                n, 1
            )  # (n * v, v)
            leace = leace.update(train_h, indicators)

        labels = repeat(train_data.labels, "n -> (n v)", v=v)

        Xs.append(train_h)
        train_labels.append(labels)

    X, train_labels = torch.cat(Xs), torch.cat(train_labels)
    eraser = leace.eraser if leace is not None else None

    if mode == "cv":
        lr_model = Classifier(X.shape[-1], device=device, eraser=eraser)
        lr_model.fit_cv(X, train_labels)
        return [lr_model]
    elif mode == "inlp":
        return Classifier.inlp(
            X, train_labels, eraser=eraser, max_iter=max_inlp_iter
        ).classifiers
    elif mode == "single":
        lr_model = Classifier(X.shape[-1], device=device, eraser=eraser)
        lr_model.fit(X, train_labels)
        return [lr_model]
    else:
        raise ValueError(f"Unknown mode: {mode}")
