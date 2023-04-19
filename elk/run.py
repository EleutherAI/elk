import os
import random
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from datasets import DatasetDict
from torch import Tensor
from tqdm import tqdm

from .debug_logging import save_debug_log
from .extraction import extract
from .files import elk_reporter_dir, memorably_named_dir
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

    def prepare_data(
        self, device: str, layer: int, split_type: Literal["train", "val"]
    ) -> dict[str, tuple[Tensor, Tensor, Tensor | None]]:
        """Prepare data for the specified layer and split type."""
        out = {}

        for ds in self.datasets:
            train_name, val_name = select_train_val_splits(ds)
            key = train_name if split_type == "train" else val_name

            split = ds[key].with_format("torch", device=device, dtype=torch.int16)
            labels = assert_type(Tensor, split["label"])
            val_h = int16_to_float32(assert_type(Tensor, split[f"hidden_{layer}"]))

            with split.formatted_as("torch", device=device):
                has_preds = "model_preds" in split.features
                lm_preds = split["model_preds"] if has_preds else None

            ds_name = get_dataset_name(ds)
            out[ds_name] = (val_h, labels, lm_preds)

        return out

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
                for df in tqdm(mapper(func, layers), total=len(layers)):
                    df_buf.append(df)
            finally:
                # Make sure the CSV is written even if we crash or get interrupted
                if df_buf:
                    df = pd.concat(df_buf).sort_values(by="layer")
                    df.to_csv(f, index=False)
                if self.cfg.debug:
                    save_debug_log(self.datasets, self.out_dir)
