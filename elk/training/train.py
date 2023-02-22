"""Main training loop."""

from ..utils import select_usable_gpus
from ..extraction import extract_hiddens, ExtractionConfig
from .preprocessing import normalize
from .reporter import OptimConfig, Reporter, ReporterConfig
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
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


def train_task(
    input_q: mp.Queue, out_q: mp.Queue, cfg: RunConfig, device: str, out_dir: Path
):
    """Worker function for training reporters in parallel."""

    # Reproducibility
    np.random.seed(cfg.net.seed)
    random.seed(cfg.net.seed)
    torch.manual_seed(cfg.net.seed)

    while not input_q.empty():
        i, *data = input_q.get()
        out_q.put((i, *train_reporter(cfg, i, *data, device, out_dir)))  # type: ignore


def train_reporter(
    cfg: RunConfig,
    layer_index: int,
    train_h: torch.Tensor,
    val_h: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    out_dir: Path,
):
    """Train a single reporter on a single layer."""

    # Note: currently we're just upcasting to float32 so we don't have to deal with
    # grad scaling (which isn't supported for LBFGS), while the hidden states are
    # saved in float16 to save disk space. In the future we could try to use mixed
    # precision training in at least some cases.
    x0, x1 = train_h.to(device).float().chunk(2, dim=-1)
    val_x0, val_x1 = val_h.to(device).float().chunk(2, dim=-1)

    train_labels_aug = torch.cat([train_labels, 1 - train_labels])
    val_labels_aug = torch.cat([val_labels, 1 - val_labels])

    reporter = Reporter(train_h.shape[-1] // 2, cfg.net, device=device)
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
        # TODO: Once we implement cross-validation for CCS, we should benchmark
        # against LogisticRegressionCV here.
        lr_model = LogisticRegression(max_iter=10_000)
        lr_model.fit(torch.cat([x0, x1]).cpu(), train_labels_aug)

        lr_preds = lr_model.predict_proba(torch.cat([val_x0, val_x1]).cpu())[:, 1]
        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)

        stats += [lr_auroc, lr_acc]
        with open(lr_dir / f"layer_{layer_index}.pkl", "wb") as file:
            pickle.dump(lr_model, file)

    with open(reporter_dir / f"layer_{layer_index}.pt", "wb") as file:
        torch.save(reporter, file)

    return stats


def train(cfg: RunConfig, out_dir: Path):
    # We use a multiprocessing context with "spawn" as the start method so CUDA works
    ctx = mp.get_context("spawn")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "cfg.yaml", "w") as f:
        cfg.dump_yaml(f)

    with open(out_dir / "model_config.json", "w") as f:
        config = AutoConfig.from_pretrained(cfg.data.model)
        json.dump(config.to_dict(), f)

    ds = DatasetDict(
        {
            split_name: Dataset.from_generator(
                extract_hiddens, gen_kwargs=dict(cfg=cfg, split=split_name)
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

    L = config.num_hidden_layers
    iterator = (ds.rename_column(f"hidden_{i}", "hiddens") for i in range(L))

    # Intelligently select device indices to use based on free memory.
    # TODO: Set the min_memory argument to some heuristic lower bound
    device_indices = select_usable_gpus(cfg.max_gpus)
    devices = [f"cuda:{i}" for i in device_indices] if device_indices else ["cpu"]

    # Collect the results and update the progress bar.
    with open(out_dir / "eval.csv", "w") as f:
        cols = ["layer", "train_loss", "loss", "acc", "cal_acc", "auroc"]
        if not cfg.skip_baseline:
            cols += ["lr_auroc", "lr_acc"]

        writer = csv.writer(f)
        writer.writerow(cols)

        # Don't bother with multiprocessing if we're only using one device.
        if len(devices) == 1:
            pbar = tqdm(iterator, total=L, unit="layer")
            for i, (train_h, val_h) in enumerate(pbar):
                train_reporter(
                    cfg,
                    i,
                    train_h,
                    val_h,
                    train_labels,
                    val_labels,
                    device=devices[0],
                    out_dir=out_dir,
                )
        else:
            # Use one queue to distribute the work to the workers, and another to
            # collect the results.
            data_queue = ctx.Queue()
            result_queue = ctx.Queue()
            for i, (train_h, val_h) in enumerate(iterator):
                data_queue.put((i, train_h, val_h, train_labels, val_labels))

            # Apparently the .put() command is non-blocking, so we need to wait
            # until the queue is filled before starting the workers.
            while data_queue.empty():
                pass

            # Start the workers.
            workers = []
            for device in devices:
                worker = ctx.Process(
                    target=train_task,
                    args=(data_queue, result_queue, cfg),
                    kwargs=dict(device=device, out_dir=out_dir),
                )
                worker.start()
                workers.append(worker)

            pbar = tqdm(total=L, unit="layer")
            for _ in range(L):
                i, *stats = result_queue.get()
                pbar.update()
                writer.writerow([L - i] + [f"{s:.4f}" for s in stats])

            # Workers should be done by now.
            pbar.close()
            for worker in workers:
                worker.join()
