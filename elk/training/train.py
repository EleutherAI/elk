"""Main training loop."""

from ..extraction import extract, ExtractionConfig
from ..files import elk_reporter_dir, memorably_named_dir
from ..utils import assert_type, held_out_split, select_usable_devices, int16_to_float32
from .classifier import Classifier
from .mlp_reporter import MlpReporter, MlpReporterConfig
from .linear_reporter import LinearReporter, LinearReporterConfig
from .preprocessing import normalize
from .reporter import OptimConfig, Reporter, ReporterConfig
from dataclasses import dataclass
from datasets import DatasetDict
from einops import rearrange
from functools import partial
from pathlib import Path
from simple_parsing import subgroups, Serializable
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor
from tqdm.auto import tqdm
from typing import cast, Literal, Optional, Union
import csv
import numpy as np
import os
import pickle
import random
import torch
import torch.multiprocessing as mp
import warnings


@dataclass
class RunConfig(Serializable):
    """Full specification of a reporter training run.

    Args:
        data: Config specifying hidden states on which the reporter will be trained.
        net: Config for building the reporter network.
        optim: Config for the `.fit()` loop.
    """

    data: ExtractionConfig
    net: ReporterConfig = subgroups(
        {"linear": LinearReporterConfig, "mlp": MlpReporterConfig},
        default=LinearReporterConfig(),  # type: ignore[arg-type]
    )
    optim: OptimConfig = OptimConfig()

    label_frac: float = 0.0
    max_gpus: int = -1
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    skip_baseline: bool = False


def train_reporter(
    cfg: RunConfig,
    dataset: DatasetDict,
    out_dir: Path,
    layer: int,
    devices: list[str],
    world_size: int = 1,
):
    """Train a single reporter on a single layer."""

    # Reproducibility
    seed = cfg.net.seed + layer
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    rank = os.getpid() % world_size
    device = devices[rank]

    # Note: currently we're just upcasting to float32 so we don't have to deal with
    # grad scaling (which isn't supported for LBFGS), while the hidden states are
    # saved in float16 to save disk space. In the future we could try to use mixed
    # precision training in at least some cases.
    with dataset.formatted_as("torch", device=device, dtype=torch.int16):
        train, val = dataset["train"], held_out_split(dataset)
        train_labels = cast(Tensor, train["label"])
        val_labels = cast(Tensor, val["label"])

        train_h, val_h = normalize(
            int16_to_float32(assert_type(Tensor, train[f"hidden_{layer}"])),
            int16_to_float32(assert_type(Tensor, val[f"hidden_{layer}"])),
            method=cfg.normalization,
        )

        x0, x1 = train_h.unbind(dim=-2)
        val_x0, val_x1 = val_h.unbind(dim=-2)

        # Check how linearly separable the pseudo-labels are. If they're very
        # separable, the algorithm may not converge to a good solution.
        pseudo_clf = Classifier(train_h.shape[-1], device=device)
        pseudo_labels = torch.cat(
            [
                torch.zeros_like(train_labels),
                torch.ones_like(train_labels),
            ]
        ).repeat_interleave(
            x0.shape[1]
        )  # make num_variants copies of each pseudo-label

        pseudo_clf.fit(
            rearrange(torch.cat([x0, x1]), "b v d -> (b v) d"), pseudo_labels
        )
        with torch.no_grad():
            pseudo_preds = pseudo_clf(
                rearrange(torch.cat([val_x0, val_x1]), "b v d -> (b v) d")
            )
            pseudo_auroc = roc_auc_score(pseudo_labels.cpu(), pseudo_preds.cpu())
            if pseudo_auroc > 0.6:
                warnings.warn(
                    f"The pseudo-labels at layer {layer} are linearly separable with "
                    f"an AUROC of {pseudo_auroc:.3f}. This may indicate that the "
                    f"algorithm will not converge to a good solution."
                )

    reporter = Reporter.instantiate(x0.shape[-1], cfg.net, device=device)
    if cfg.label_frac:
        num_labels = round(cfg.label_frac * len(train_labels))
        labels = train_labels[:num_labels]
    else:
        labels = None

    train_loss = reporter.fit((x0, x1), labels, cfg.optim)
    val_results = reporter.score(
        (val_x0, val_x1),
        val_labels,
    )

    lr_dir = out_dir / "lr_models"
    reporter_dir = out_dir / "reporters"

    lr_dir.mkdir(parents=True, exist_ok=True)
    reporter_dir.mkdir(parents=True, exist_ok=True)
    stats = [layer, pseudo_auroc, train_loss, *val_results]

    if not cfg.skip_baseline:
        # repeat_interleave makes `num_variants` copies of each label, all within a
        # single dimension of size `num_variants * 2 * n`, such that the labels align
        # with X.view(-1, X.shape[-1])
        train_labels_aug = torch.cat(
            [train_labels, 1 - train_labels]
        ).repeat_interleave(x0.shape[1])
        val_labels_aug = (
            torch.cat([val_labels, 1 - val_labels]).repeat_interleave(x0.shape[1])
        ).cpu()

        X = torch.cat([x0, x1]).squeeze()
        d = X.shape[-1]
        lr_model = Classifier(d, device=device)
        lr_model.fit(X.view(-1, d), train_labels_aug)

        X_val = torch.cat([val_x0, val_x1]).view(-1, d)
        with torch.no_grad():
            lr_preds = lr_model(X_val).sigmoid().cpu()

        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)

        stats += [lr_auroc, lr_acc]
        with open(lr_dir / f"layer_{layer}.pkl", "wb") as file:
            pickle.dump(lr_model, file)

    with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
        torch.save(reporter, file)

    return stats


def train(cfg: RunConfig, out_dir: Optional[Path] = None):
    # Extract the hidden states first if necessary
    ds = extract(cfg.data, max_gpus=cfg.max_gpus)

    if out_dir is None:
        out_dir = memorably_named_dir(elk_reporter_dir())
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print the output directory in bold with escape codes
    print(f"Saving results to \033[1m{out_dir}\033[0m")

    with open(out_dir / "cfg.yaml", "w") as f:
        cfg.dump_yaml(f)

    devices = select_usable_devices(cfg.max_gpus)
    num_devices = len(devices)

    cols = ["layer", "pseudo_auroc", "train_loss", "loss", "acc", "cal_acc", "auroc"]
    if not cfg.skip_baseline:
        cols += ["lr_auroc", "lr_acc"]

    layers = [
        int(feat[len("hidden_") :])
        for feat in ds["train"].features
        if feat.startswith("hidden_")
    ]
    # Train reporters for each layer in parallel
    with mp.Pool(num_devices) as pool, open(out_dir / "eval.csv", "w") as f:
        fn = partial(
            train_reporter, cfg, ds, out_dir, devices=devices, world_size=num_devices
        )
        writer = csv.writer(f)
        writer.writerow(cols)

        mapper = pool.imap if num_devices > 1 else map
        for i, *stats in tqdm(mapper(fn, layers), total=len(layers)):
            writer.writerow([i] + [f"{s:.4f}" for s in stats])
