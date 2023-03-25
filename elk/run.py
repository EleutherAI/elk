import csv
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Iterator, TextIO, Callable, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import DatasetDict
from torch import Tensor
from tqdm import tqdm

from elk.extraction.extraction import extract
from elk.files import create_output_directory, save_config, save_meta
from elk.logging import save_debug_log
from elk.training.preprocessing import normalize
from elk.training.train_result import ElicitStatResult, EvalStatResult
from elk.utils.data_utils import get_layers, select_train_val_splits
from elk.utils.gpu_utils import select_usable_devices
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

    @abstractmethod
    def apply_to_single_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int = 1,
    ) -> Union[ElicitStatResult, EvalStatResult]:
        ...

    def apply_to_layers(self):
        """Apply a function to each layer of the dataset in parallel."""
        func = self.apply_to_single_layer

        devices = select_usable_devices(self.cfg.num_gpus)
        num_devices = len(devices)
        self.out_dir = assert_type(Path, self.out_dir)
        with mp.Pool(num_devices) as pool, open(self.out_dir / "eval.csv", "w") as f:
            # Partially apply so the function will just take the layer as an argument
            fn: Callable[[int], Union[ElicitStatResult, EvalStatResult]] = partial(
                func, devices=devices, world_size=num_devices
            )
            layers: list[int] = get_layers(self.dataset)
            mapper = pool.imap_unordered if num_devices > 1 else map
            # Typed as sequence for covariant typing
            iterator: Sequence[Union[ElicitStatResult, EvalStatResult]] = tqdm(mapper(fn, layers), total=len(layers))  # type: ignore
            write_func_to_file(
                iterator=iterator,
                file=f,
                debug=self.cfg.debug,
                dataset=self.dataset,
                out_dir=self.out_dir,
                skip_baseline=self.cfg.skip_baseline,
            )


def write_func_to_file(
    iterator: Sequence[Union[ElicitStatResult, EvalStatResult]],
    file: TextIO,
    debug: bool,
    dataset: DatasetDict,
    out_dir: Path,
    skip_baseline: bool,
) -> None:
    row_buf = []
    writer = csv.writer(file)
    # write a single line
    writer.writerow(ElicitStatResult.to_csv_columns(skip_baseline=skip_baseline))
    try:
        for row in iterator:
            row_buf.append(row)
    finally:
        # Make sure the CSV is written even if we crash or get interrupted
        for row in sorted(row_buf):
            writer.writerow(row)
        if debug:
            save_debug_log(dataset, out_dir)
