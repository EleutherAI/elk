import os
import random
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
    Callable,
    Iterator,
)

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import DatasetDict
from torch import Tensor
from tqdm import tqdm

from elk.extraction.extraction import extract
from elk.files import create_output_directory, save_config, save_meta
from elk.training.preprocessing import normalize
from elk.utils.csv import write_iterator_to_file, Log
from elk.utils.data_utils import get_layers, select_train_val_splits
from elk.utils.typing import assert_type, int16_to_float32

if TYPE_CHECKING:
    from elk.evaluation.evaluate import Eval
    from elk.training.train import Elicit


@dataclass
class Run(ABC):
    cfg: Union["Elicit", "Eval"]
    out_dir: Optional[Path] = None
    dataset: DatasetDict = field(init=False)

    def __post_init__(self):
        # Extract the hidden states first if necessary
        self.dataset = extract(self.cfg.data, num_gpus=self.cfg.num_gpus)

        self.out_dir = create_output_directory(self.out_dir)
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

            x0, x1 = train_h.unbind(dim=-2)
            val_x0, val_x1 = val_h.unbind(dim=-2)

        return x0, x1, val_x0, val_x1, train_labels, val_labels

    def concatenate(self, layers):
        """Concatenate hidden states from a previous layer."""
        for layer in range(self.cfg.concatenated_layer_offset, len(layers)):
            layers[layer] = layers[layer] + [
                layers[layer][0] - self.cfg.concatenated_layer_offset
            ]
        return layers

    def apply_to_layers(
        self,
        func: Callable[[int], Log],
        num_devices: int,
        to_csv_line: Callable[[Log], list[str]],
        csv_columns: list[str],
    ):
        """Apply a function to each layer of the dataset in parallel
        and writes the results to a CSV file.

        Args:
            func: The function to apply to each layer.
                The int is the index of the layer.
            num_devices: The number of devices to use.
            to_csv_line: A function that converts a Log to a list of strings.
                This has to be injected in because the Run class does not know
                the extra options e.g. skip_baseline to apply to function.
            csv_columns: The columns of the CSV file."""
        self.out_dir = assert_type(Path, self.out_dir)

        layers: list[int] = get_layers(self.dataset)

        if self.cfg.concatenated_layer_offset > 0:
            layers = self.concatenate(layers)

        # Should we write to different CSV files for elicit vs eval?
        # set spawn start method to avoid issues with CUDA
        mp.set_start_method("spawn", force=True)
        with mp.Pool(num_devices) as pool, open(self.out_dir / "eval.csv", "w") as f:
            mapper = pool.imap_unordered if num_devices > 1 else map
            iterator: Iterator[Log] = tqdm(  # type: ignore
                mapper(func, layers), total=len(layers)
            )
            write_iterator_to_file(
                iterator=iterator,
                file=f,
                debug=self.cfg.debug,
                dataset=self.dataset,
                out_dir=self.out_dir,
                csv_columns=csv_columns,
                to_csv_line=to_csv_line,
            )
