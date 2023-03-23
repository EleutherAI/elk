import csv
import os
import pickle
from dataclasses import dataclass
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import List, Literal, Optional, cast

import torch
import torch.multiprocessing as mp
import yaml
from simple_parsing.helpers import Serializable, field
from torch import Tensor
from tqdm.auto import tqdm

from datasets import DatasetDict
from elk.training.preprocessing import normalize

from ..extraction import ExtractionConfig, extract
from ..files import elk_reporter_dir, memorably_named_dir
from ..utils import (
    assert_type,
    int16_to_float32,
    select_train_val_splits,
    select_usable_devices,
)


@dataclass
class EvaluateConfig(Serializable):
    target: ExtractionConfig
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    num_gpus: int = -1


def evaluate_reporter(
    cfg: EvaluateConfig,
    dataset: DatasetDict,
    layer: int,
    devices: list[str],
    world_size: int = 1,
):
    """Evaluate a single reporter on a single layer."""
    rank = os.getpid() % world_size
    device = devices[rank]

    # Note: currently we're just upcasting to float32 so we don't have to deal with
    # grad scaling (which isn't supported for LBFGS), while the hidden states are
    # saved in float16 to save disk space. In the future we could try to use mixed
    # precision training in at least some cases.
    with dataset.formatted_as("torch", device=device, dtype=torch.int16):
        train_split, val_split = select_train_val_splits(dataset)
        train, val = dataset[train_split], dataset[val_split]
        test_labels = cast(Tensor, val["label"])

        _, test_h = normalize(
            int16_to_float32(assert_type(Tensor, train[f"hidden_{layer}"])),
            int16_to_float32(assert_type(Tensor, val[f"hidden_{layer}"])),
            method=cfg.normalization,
        )

    reporter_path = elk_reporter_dir() / cfg.source / "reporters" / f"layer_{layer}.pt"
    reporter = torch.load(reporter_path, map_location=device)
    reporter.eval()

    test_x0, test_x1 = test_h.unbind(dim=-2)

    test_result = reporter.score(test_labels, test_x0, test_x1)

    stats = [layer, *test_result]
    return stats


def evaluate_reporters(cfg: EvaluateConfig, out_dir: Optional[Path] = None):
    ds = extract(cfg.target, num_gpus=cfg.num_gpus)

    layers = [
        int(feat[len("hidden_") :])
        for feat in ds["train"].features
        if feat.startswith("hidden_")
    ]

    devices = select_usable_devices(cfg.num_gpus)
    num_devices = len(devices)

    transfer_eval = elk_reporter_dir() / cfg.source / "transfer_eval"
    transfer_eval.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        out_dir = memorably_named_dir(transfer_eval)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print the output directory in bold with escape codes
    print(f"Saving results to \033[1m{out_dir}\033[0m")

    with open(out_dir / "cfg.yaml", "w") as f:
        cfg.dump_yaml(f)

    cols = ["layer", "loss", "acc", "cal_acc", "auroc"]
    # Evaluate reporters for each layer in parallel
    with mp.Pool(num_devices) as pool, open(out_dir / "eval.csv", "w") as f:
        fn = partial(
            evaluate_reporter, cfg, ds, devices=devices, world_size=num_devices
        )
        writer = csv.writer(f)
        writer.writerow(cols)

        mapper = pool.imap_unordered if num_devices > 1 else map
        row_buf = []
        try:
            for i, *stats in tqdm(mapper(fn, layers), total=len(layers)):
                row_buf.append([i] + [f"{s:.4f}" for s in stats])
        finally:
            # Make sure the CSV is written even if we crash or get interrupted
            for row in sorted(row_buf):
                writer.writerow(row)

    print("Results saved")
