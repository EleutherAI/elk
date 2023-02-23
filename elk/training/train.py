"""Main training loop."""

from ..utils import select_usable_gpus
from ..extraction import extract_hiddens, ExtractionConfig
from .preprocessing import normalize
from .reporter import OptimConfig, Reporter, ReporterConfig
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from functools import partial
from pathlib import Path
from simple_parsing import Serializable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
from transformers import AutoConfig
from typing import Literal
import csv
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.multiprocessing as mp


@dataclass
class RunConfig(Serializable):
    """Full specification of a reporter training run.

    Args:
        data: Config specifying hidden states on which the reporter will be trained.
        net: Config for building the reporter network.
        optim: Config for the `.fit()` loop.
    """

    data: ExtractionConfig
    net: ReporterConfig
    optim: OptimConfig

    label_frac: float = 0.0
    max_gpus: int = -1
    normalization: Literal["legacy", "elementwise", "meanonly"] = "meanonly"
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
    with dataset.formatted_as("torch"):
        train, val = dataset["train"], dataset["validation"]

        x0, x1 = torch.stack(train[f"hidden_{layer}"]).float().chunk(2, dim=-1)
        val_x0, val_x1 = torch.stack(val[f"hidden_{layer}"]).float().chunk(2, dim=-1)
        train_labels = torch.stack(train["label"])
        val_labels = torch.stack(val["label"])

    reporter = Reporter(x0.shape[-1], cfg.net, device=device)
    if cfg.label_frac:
        num_labels = round(cfg.label_frac * len(train_labels))
        labels = train_labels[:num_labels].to(device)
    else:
        labels = None

    train_loss = reporter.fit((x0, x1), labels, cfg.optim)
    val_result = reporter.score(
        (val_x0, val_x1),
        val_labels.to(device),
    )

    lr_dir = out_dir / "lr_models"
    reporter_dir = out_dir / "reporters"

    lr_dir.mkdir(parents=True, exist_ok=True)
    reporter_dir.mkdir(parents=True, exist_ok=True)
    stats = [train_loss, *val_result]

    if not cfg.skip_baseline:
        train_labels_aug = torch.cat([train_labels, 1 - train_labels])
        val_labels_aug = torch.cat([val_labels, 1 - val_labels])

        # TODO: Once we implement cross-validation for CCS, we should benchmark
        # against LogisticRegressionCV here.
        lr_model = LogisticRegression(max_iter=10_000)
        lr_model.fit(torch.cat([x0, x1]).cpu(), train_labels_aug)

        lr_preds = lr_model.predict_proba(torch.cat([val_x0, val_x1]).cpu())[:, 1]
        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)

        stats += [lr_auroc, lr_acc]
        with open(lr_dir / f"layer_{layer}.pkl", "wb") as file:
            pickle.dump(lr_model, file)

    with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
        torch.save(reporter, file)

    return stats


def train(cfg: RunConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cfg.yaml", "w") as f:
        cfg.dump_yaml(f)

    with open(out_dir / "model_config.json", "w") as f:
        config = AutoConfig.from_pretrained(cfg.data.model)
        json.dump(config.to_dict(), f)

    ds = DatasetDict(
        {
            split_name: Dataset.from_generator(
                extract_hiddens, gen_kwargs=dict(cfg=cfg.data, split=split_name)
            )
            for split_name in ["train", "validation"]
        }
    )
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Re-implement this in a way that doesn't require loading all the hidden
    # states into memory at once.
    # train_hiddens, val_hiddens = normalize(
    #     train_hiddens, val_hiddens, cfg.normalization
    # )

    # Intelligently select device indices to use based on free memory.
    # TODO: Set the min_memory argument to some heuristic lower bound
    gpu_indices = select_usable_gpus(cfg.max_gpus)
    devices = [f"cuda:{i}" for i in gpu_indices] if gpu_indices else ["cpu"]
    num_devices = len(devices)

    cols = ["layer", "train_loss", "loss", "acc", "cal_acc", "auroc"]
    if not cfg.skip_baseline:
        cols += ["lr_auroc", "lr_acc"]

    # Train reporters for each layer in parallel
    with mp.Pool(num_devices) as pool, open(out_dir / "eval.csv", "w") as f:
        fn = partial(
            train_reporter, cfg, ds, out_dir, devices=devices, world_size=num_devices
        )
        writer = csv.writer(f)
        writer.writerow(cols)

        L = config.num_hidden_layers
        for i, *stats in tqdm(pool.imap_unordered(fn, range(L))):
            writer.writerow([i] + [f"{s:.4f}" for s in stats])
