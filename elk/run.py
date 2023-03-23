import csv
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Union, cast, TYPE_CHECKING

import numpy as np
import torch
import torch.multiprocessing as mp

from torch import Tensor
from tqdm import tqdm

from datasets import DatasetDict, Split

from elk.extraction.extraction import extract
from elk.files import create_output_directory, save_config
from elk.training.preprocessing import normalize
from elk.utils.data_utils import get_layers, select_train_val_splits
from elk.utils.gpu_utils import select_usable_devices
from elk.utils.typing import upcast_hiddens

if TYPE_CHECKING:
    from elk.training.train import Elicit
    from elk.evaluation.evaluate import Eval


@dataclass
class Run(ABC):
    cfg: Union["Elicit", "Eval"]

    def make_reproducible(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def get_device(self, devices: List[str], world_size: int):
        rank = os.getpid() % world_size
        device = devices[rank]
        return device

    def prepare_data(
        self,
        dataset: DatasetDict,
        device: str,
        layer: int,
        priorities: dict = {Split.TRAIN: 0, Split.VALIDATION: 1, Split.TEST: 2},
    ):
        with dataset.formatted_as("torch", device=device, dtype=torch.int16):
            train_split, val_split = select_train_val_splits(
                dataset, priorities=priorities
            )
            train, val = dataset[train_split], dataset[val_split]

            train_labels = cast(Tensor, train["label"])
            val_labels = cast(Tensor, val["label"])

            train_h, val_h = normalize(
                upcast_hiddens(train[f"hidden_{layer}"]),  # type: ignore
                upcast_hiddens(val[f"hidden_{layer}"]),  # type: ignore
                method=self.cfg.normalization,
            )

            x0, x1 = train_h.unbind(dim=-2)
            val_x0, val_x1 = val_h.unbind(dim=-2)

        return x0, x1, val_x0, val_x1, train_labels, val_labels

    def run_on_layers(
        self,
        func: Callable,
        cols: List[str],
        out_dir: Path,
        cfg: Union["Elicit", "Eval"],
        ds,
        layers: List[int],
    ):
        devices = select_usable_devices(cfg.max_gpus)
        num_devices = len(devices)

        with mp.Pool(num_devices) as pool, open(out_dir / "eval.csv", "w") as f:
            fn = partial(func, ds, out_dir, devices=devices, world_size=num_devices)
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

    def run(self, func: Callable, cols: List[str], out_dir: Optional[Path] = None):
        # Extract the hidden states first if necessary
        ds = extract(self.cfg.data, max_gpus=self.cfg.max_gpus)

        out_dir = create_output_directory(out_dir)
        save_config(self.cfg, out_dir)

        self.run_on_layers(
            func=func,
            cols=cols,
            out_dir=out_dir,
            cfg=self.cfg,
            ds=ds,
            layers=get_layers(ds),
        )
