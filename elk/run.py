import os
import random
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from datasets import DatasetDict, concatenate_datasets
from torch import Tensor
from tqdm import tqdm

from .extraction import extract
from .files import elk_reporter_dir, memorably_named_dir
from .logging import save_debug_log
from .utils import (
    assert_type,
    get_dataset_name,
    get_layers,
    int16_to_float32,
    select_train_val_splits,
)

if TYPE_CHECKING:
    from .evaluation.evaluate import Eval
    from .training.train import Elicit


@dataclass
class Run(ABC):
    cfg: Union["Elicit", "Eval"]
    out_dir: Path | None = None
    datasets: list[DatasetDict] = field(init=False)

    def __post_init__(self):
        self.datasets = [
            extract(cfg, num_gpus=self.cfg.num_gpus, min_gpu_mem=self.cfg.min_gpu_mem)
            for cfg in self.cfg.data.explode()
        ]

        if self.out_dir is None:
            # Save in a memorably-named directory inside of
            # ELK_REPORTER_DIR/<model_name>/<dataset_name>
            ds_name = ", ".join(self.cfg.data.prompts.datasets)
            root = elk_reporter_dir() / self.cfg.data.model / ds_name

            self.out_dir = memorably_named_dir(root)

        # Print the output directory in bold with escape codes
        print(f"Output directory at \033[1m{self.out_dir}\033[0m")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        path = self.out_dir / "cfg.yaml"
        with open(path, "w") as f:
            self.cfg.dump_yaml(f)

        path = self.out_dir / "fingerprints.yaml"
        with open(path, "w") as meta_f:
            yaml.dump(
                {
                    get_dataset_name(ds): {
                        split: ds[split]._fingerprint for split in ds.keys()
                    }
                    for ds in self.datasets
                },
                meta_f,
            )

    def make_reproducible(self, seed: int):
        """Make the run reproducible by setting the random seed."""

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def get_device(self, devices, world_size: int) -> str:
        """Get the device for the current process."""

        rank = os.getpid() % world_size
        device = devices[rank]
        return device

    def prepare_data(self, device: str, layer: int) -> tuple:
        """Prepare the data for training and validation."""

        train_sets = []
        val_output = {}

        # We handle train and val differently. We want to concatenate all of the
        # train sets together, but we want to keep the val sets separate so that we can
        # compute evaluation metrics separately for each dataset.
        for ds in self.datasets:
            train_split, val_split = select_train_val_splits(ds)
            train_sets.append(ds[train_split])

            val = ds[val_split].with_format("torch", device=device, dtype=torch.int16)
            val_labels = assert_type(Tensor, val["label"])
            val_h = int16_to_float32(assert_type(torch.Tensor, val[f"hidden_{layer}"]))
            val_x0, val_x1 = val_h.unbind(dim=-2)

            with val.formatted_as("numpy"):
                has_preds = "model_preds" in val.features
                val_lm_preds = val["model_preds"] if has_preds else None

            ds_name = get_dataset_name(ds)
            val_output[ds_name] = (val_x0, val_x1, val_labels, val_lm_preds)

        train = concatenate_datasets(train_sets).with_format(
            "torch", device=device, dtype=torch.int16
        )

        train_labels = assert_type(Tensor, train["label"])
        train_h = int16_to_float32(assert_type(torch.Tensor, train[f"hidden_{layer}"]))
        x0, x1 = train_h.unbind(dim=-2)

        return x0, x1, train_labels, val_output

    def concatenate(self, layers):
        """Concatenate hidden states from a previous layer."""
        for layer in range(self.cfg.concatenated_layer_offset, len(layers)):
            layers[layer] += [layers[layer][0] - self.cfg.concatenated_layer_offset]

        return layers

    def apply_to_layers(
        self,
        func: Callable[[int], pd.DataFrame],
        num_devices: int,
    ):
        """Apply a function to each layer of the datasets in parallel
        and writes the results to a CSV file.

        Args:
            func: The function to apply to each layer.
                The int is the index of the layer.
            num_devices: The number of devices to use.
        """
        self.out_dir = assert_type(Path, self.out_dir)

        layers, *rest = [get_layers(ds) for ds in self.datasets]
        assert all(x == layers for x in rest), "All datasets must have the same layers"

        if self.cfg.concatenated_layer_offset > 0:
            layers = self.concatenate(layers)

        # Should we write to different CSV files for elicit vs eval?
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_devices) as pool, open(self.out_dir / "eval.csv", "w") as f:
            mapper = pool.imap_unordered if num_devices > 1 else map
            df_buf = []

            try:
                for row in tqdm(mapper(func, layers), total=len(layers)):
                    df_buf.append(row)
            finally:
                # Make sure the CSV is written even if we crash or get interrupted
                if df_buf:
                    df = pd.concat(df_buf).sort_values(by="layer")
                    df.to_csv(f, index=False)
                if self.cfg.debug:
                    save_debug_log(self.datasets, self.out_dir)
