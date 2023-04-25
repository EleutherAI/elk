import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from simple_parsing.helpers import Serializable, field
from torch import Tensor
from tqdm import tqdm

from .debug_logging import save_debug_log
from .extraction import Extract, extract
from .extraction.dataset_name import DatasetDictWithName
from .files import elk_reporter_dir, memorably_named_dir
from .utils import (
    assert_type,
    get_layers,
    int16_to_float32,
    select_train_val_splits,
    select_usable_devices,
)


@dataclass
class Run(ABC, Serializable):
    data: Extract
    out_dir: Path | None = None
    """Directory to save results to. If None, a directory will be created
    automatically."""

    datasets: list[DatasetDictWithName] = field(default_factory=list, init=False)
    """Datasets containing hidden states and labels for each layer."""

    concatenated_layer_offset: int = 0
    debug: bool = False
    min_gpu_mem: int | None = None
    num_gpus: int = -1
    out_dir: Path | None = None
    disable_cache: bool = field(default=False, to_dict=False)

    def execute(self, highlight_color: str = "cyan"):
        self.datasets = [
            extract(
                cfg,
                disable_cache=self.disable_cache,
                highlight_color=highlight_color,
                num_gpus=self.num_gpus,
                min_gpu_mem=self.min_gpu_mem,
            )
            for cfg in self.data.explode()
        ]

        if self.out_dir is None:
            # Save in a memorably-named directory inside of
            # ELK_REPORTER_DIR/<model_name>/<dataset_name>
            ds_name = ", ".join(self.data.prompts.datasets)
            root = elk_reporter_dir() / self.data.model / ds_name

            self.out_dir = memorably_named_dir(root)

        # Print the output directory in bold with escape codes
        print(f"Output directory at \033[1m{self.out_dir}\033[0m")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        path = self.out_dir / "cfg.yaml"
        with open(path, "w") as f:
            self.dump_yaml(f)

        path = self.out_dir / "fingerprints.yaml"
        with open(path, "w") as meta_f:
            yaml.dump(
                {
                    ds_name: {
                        split: ds[split]._fingerprint for split in ds.keys()
                    }
                    for ds_name, ds in self.datasets
                },
                meta_f,
            )

        devices = select_usable_devices(self.num_gpus, min_memory=self.min_gpu_mem)
        num_devices = len(devices)
        func: Callable[[int], pd.DataFrame] = partial(
            self.apply_to_layer, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)

    @abstractmethod
    def apply_to_layer(
        self, layer: int, devices: list[str], world_size: int
    ) -> pd.DataFrame:
        """Train or eval a reporter on a single layer."""

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

        for ds_name, ds in self.datasets:
            train_name, val_name = select_train_val_splits(ds)
            key = train_name if split_type == "train" else val_name

            split = ds[key].with_format("torch", device=device, dtype=torch.int16)
            labels = assert_type(Tensor, split["label"])
            val_h = int16_to_float32(assert_type(Tensor, split[f"hidden_{layer}"]))

            with split.formatted_as("torch", device=device):
                has_preds = "model_logits" in split.features
                lm_preds = split["model_logits"] if has_preds else None

            out[ds_name] = (val_h, labels.to(val_h.device), lm_preds)

        return out

    def concatenate(self, layers):
        """Concatenate hidden states from a previous layer."""
        for layer in range(self.concatenated_layer_offset, len(layers)):
            layers[layer] += [layers[layer][0] - self.concatenated_layer_offset]

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

        layers, *rest = [get_layers(ds) for _, ds in self.datasets]
        assert all(x == layers for x in rest), "All datasets must have the same layers"

        if self.concatenated_layer_offset > 0:
            layers = self.concatenate(layers)

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
                    df.round(4).to_csv(f, index=False)
                if self.debug:
                    save_debug_log(self.datasets, self.out_dir)
