import os
import random
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from datasets import DatasetDict
from torch import Tensor
from tqdm import tqdm

from .extraction import extract
from .files import elk_reporter_dir, memorably_named_dir, save_config, save_meta
from .logging import save_debug_log
from .training.preprocessing import normalize
from .utils import assert_type, int16_to_float32
from .utils.data_utils import get_layers, select_train_val_splits

if TYPE_CHECKING:
    from .evaluation.evaluate import Eval
    from .training.train import Elicit


@dataclass
class Run(ABC):
    cfg: Union["Elicit", "Eval"]
    out_dir: Optional[Path] = None
    dataset: DatasetDict = field(init=False)

    def __post_init__(self):
        # Extract the hidden states first if necessary
        self.dataset = extract(self.cfg.data, num_gpus=self.cfg.num_gpus)

        if self.out_dir is None:
            # Save in a memorably-named directory inside of
            # ELK_REPORTER_DIR/<model_name>/<dataset_name>
            ds_name = ", ".join(self.cfg.data.prompts.datasets)
            root = elk_reporter_dir() / self.cfg.data.model / ds_name

            self.out_dir = memorably_named_dir(root)

        # Print the output directory in bold with escape codes
        print(f"Output directory at \033[1m{self.out_dir}\033[0m")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        save_config(self.cfg, self.out_dir)
        save_meta(self.dataset, self.out_dir)

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
        self,
        device: str,
        layer: int,
    ) -> tuple:
        """Prepare the data for training and validation."""

        with self.dataset.formatted_as("torch", device=device, dtype=torch.int16):
            train_split, val_split = select_train_val_splits(self.dataset)
            train, val = self.dataset[train_split], self.dataset[val_split]

            train_labels = assert_type(Tensor, train["label"])
            val_labels = assert_type(Tensor, val["label"])

            # Note: currently we're just upcasting to float32
            # so we don't have to deal with
            # grad scaling (which isn't supported for LBFGS),
            # while the hidden states are
            # saved in float16 to save disk space.
            # In the future we could try to use mixed
            # precision training in at least some cases.
            train_h, val_h = normalize(
                int16_to_float32(assert_type(torch.Tensor, train[f"hidden_{layer}"])),
                int16_to_float32(assert_type(torch.Tensor, val[f"hidden_{layer}"])),
                method=self.cfg.normalization,
            )

        with self.dataset.formatted_as("torch"):
            has_preds = "model_preds" in val.features
            val_lm_preds = val["model_preds"] if has_preds else None

        return train_h, val_h, train_labels, val_labels, val_lm_preds

    def concatenate(self, layers):
        """Concatenate hidden states from a previous layer."""
        for layer in range(self.cfg.concatenated_layer_offset, len(layers)):
            layers[layer] = layers[layer] + [
                layers[layer][0] - self.cfg.concatenated_layer_offset
            ]
        return layers

    def apply_to_layers(
        self,
        func: Callable[[int], pd.Series],
        num_devices: int,
    ):
        """Apply a function to each layer of the dataset in parallel
        and writes the results to a CSV file.

        Args:
            func: The function to apply to each layer.
                The int is the index of the layer.
            num_devices: The number of devices to use.
        """
        self.out_dir = assert_type(Path, self.out_dir)

        layers: list[int] = get_layers(self.dataset)

        if self.cfg.concatenated_layer_offset > 0:
            layers = self.concatenate(layers)

        # Should we write to different CSV files for elicit vs eval?
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_devices) as pool, open(self.out_dir / "eval.csv", "w") as f:
            mapper = pool.imap_unordered if num_devices > 1 else map
            row_buf = []

            try:
                for row in tqdm(mapper(func, layers), total=len(layers)):
                    row_buf.append(row)
            finally:
                # Make sure the CSV is written even if we crash or get interrupted
                if row_buf:
                    df = pd.DataFrame(row_buf).sort_values(by="layer")
                    df.to_csv(f, index=False)
                if self.cfg.debug:
                    save_debug_log(self.dataset, self.out_dir)
