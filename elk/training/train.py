"""Main training loop."""

from elk.extraction.extraction import extract
from elk.files import create_output_directory, save_config
import elk.parallelization
from elk.utils.typing import upcast_hiddens
from ..extraction import ExtractionConfig
from .classifier import Classifier
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .preprocessing import normalize
from .reporter import OptimConfig, Reporter, ReporterConfig
from dataclasses import dataclass
from datasets import DatasetDict
from pathlib import Path
from simple_parsing import field, subgroups, Serializable
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor
from typing import cast, Literal, Optional
import numpy as np
import os
import pickle
import random
import torch
import warnings
from elk.utils.data_utils import get_layers, select_train_val_splits
from elk.parallelization import run_on_layers

@dataclass
class TrainConfig(Serializable):
    """Full specification of a reporter training run.

    Args:
        data: Config specifying hidden states on which the reporter will be trained.
        net: Config for building the reporter network.
        optim: Config for the `.fit()` loop.
    """

    data: ExtractionConfig
    net: ReporterConfig = subgroups(
        {"ccs": CcsReporterConfig, "eigen": EigenReporterConfig}, default="eigen"
    )
    optim: OptimConfig = field(default_factory=OptimConfig)

    label_frac: float = 0.0
    max_gpus: int = -1
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    skip_baseline: bool = False


def train_reporter(
    cfg: TrainConfig,
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
        train_split, val_split = select_train_val_splits(dataset)
        train, val = dataset[train_split], dataset[val_split]

        train_labels = cast(Tensor, train["label"])
        val_labels = cast(Tensor, val["label"])

        train_h, val_h = normalize(
            upcast_hiddens(train[f"hidden_{layer}"]), # type: ignore
            upcast_hiddens(val[f"hidden_{layer}"]), # type: ignore
            method=cfg.normalization,
        )

        x0, x1 = train_h.unbind(dim=-2)
        val_x0, val_x1 = val_h.unbind(dim=-2)

        with torch.no_grad():
            pseudo_auroc = Reporter.check_separability(
                train_pair=(x0, x1), val_pair=(val_x0, val_x1)
            )
            if pseudo_auroc > 0.6:
                warnings.warn(
                    f"The pseudo-labels at layer {layer} are linearly separable with "
                    f"an AUROC of {pseudo_auroc:.3f}. This may indicate that the "
                    f"algorithm will not converge to a good solution."
                )

    if isinstance(cfg.net, CcsReporterConfig):
        reporter = CcsReporter(x0.shape[-1], cfg.net, device=device)
    elif isinstance(cfg.net, EigenReporterConfig):
        reporter = EigenReporter(x0.shape[-1], cfg.net, device=device)
    else:
        raise ValueError(f"Unknown reporter config type: {type(cfg.net)}")

    train_loss = reporter.fit(x0, x1, train_labels)
    val_result = reporter.score(
        val_labels,
        val_x0,
        val_x1,
    )

    lr_dir = out_dir / "lr_models"
    reporter_dir = out_dir / "reporters"

    lr_dir.mkdir(parents=True, exist_ok=True)
    reporter_dir.mkdir(parents=True, exist_ok=True)
    stats = [layer, pseudo_auroc, train_loss, *val_result]

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
        with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
            pickle.dump(lr_model, file)

    with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
        torch.save(reporter, file)

    return stats


def train(cfg: TrainConfig, out_dir: Optional[Path] = None):
    cols = [
        "layer",
        "pseudo_auroc",
        "train_loss",
        "acc",
        "cal_acc",
        "auroc",
        "ece",
    ]
    if not cfg.skip_baseline:
        cols += ["lr_auroc", "lr_acc"]

    # Extract the hidden states first if necessary
    ds = extract(cfg.data, max_gpus=cfg.max_gpus)
    
    out_dir = create_output_directory(out_dir)
    save_config(cfg, out_dir)

    run_on_layers(
        func=train_reporter,
        cols=cols,
        out_dir=out_dir,
        cfg=cfg,
        ds=ds,
        layers=get_layers(ds)
    )