import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
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
from simple_parsing.helpers.serialization import save
from torch import Tensor
from tqdm import tqdm

from .debug_logging import save_debug_log
from .extraction import Extract, extract
from .extraction.dataset_name import DatasetDictWithName
from .files import elk_reporter_dir, memorably_named_dir
from .metrics.eval import LayerOutput, layer_ensembling
from .utils import (
    Color,
    assert_type,
    get_layer_indices,
    int16_to_float32,
    select_split,
    select_usable_devices,
)
from .utils.types import PromptEnsembling

PROMPT_ENSEMBLING = "prompt_ensembling"


@dataclass(frozen=True)
class LayerApplied:
    layer_outputs: list[LayerOutput]
    """The output of the reporter on the layer, should contain credences and ground
    truth labels."""
    df_dict: dict[str, pd.DataFrame]
    """The evaluation results for the layer."""


def calculate_layer_outputs(layer_outputs: list[LayerOutput], out_path: Path):
    """
    Calculate the layer ensembling results for each dataset
    and prompt ensembling and save them to a CSV file.

    Args:
        layer_outputs: The layer outputs to calculate the results for.
        out_path: The path to save the results to.
    """
    grouped_layer_outputs = {}
    for layer_output in layer_outputs:
        dataset_name = layer_output.meta["dataset"]
        if dataset_name in grouped_layer_outputs:
            grouped_layer_outputs[dataset_name].append(layer_output)
        else:
            grouped_layer_outputs[dataset_name] = [layer_output]

    dfs = []
    for dataset_name, layer_outputs in grouped_layer_outputs.items():
        for prompt_ensembling in PromptEnsembling.all():
            res = layer_ensembling(
                layer_outputs=layer_outputs,
                prompt_ensembling=prompt_ensembling,
            )
            df = pd.DataFrame(
                {
                    "dataset": dataset_name,
                    PROMPT_ENSEMBLING: prompt_ensembling.value,
                    **res.to_dict(),
                },
                index=[0],
            ).round(4)
            dfs.append(df)

    df_concat = pd.concat(dfs)
    df_concat.to_csv(out_path, index=False)


@dataclass
class Run(ABC, Serializable):
    data: Extract
    out_dir: Path | None = None
    """Directory to save results to. If None, a directory will be created
    automatically."""

    datasets: list[DatasetDictWithName] = field(
        default_factory=list, init=False, to_dict=False
    )
    """Datasets containing hidden states and labels for each layer."""

    prompt_indices: tuple[int, ...] = ()
    """The indices of the prompt templates to use. If empty, all prompts are used."""

    concatenated_layer_offset: int = 0
    debug: bool = False
    min_gpu_mem: int | None = None  # in bytes
    num_gpus: int = -1
    out_dir: Path | None = None
    disable_cache: bool = field(default=False, to_dict=False)

    def execute(
        self,
        highlight_color: Color = "cyan",
        split_type: Literal["train", "val", None] = None,
    ):
        self.datasets = [
            extract(
                cfg,
                disable_cache=self.disable_cache,
                highlight_color=highlight_color,
                num_gpus=self.num_gpus,
                min_gpu_mem=self.min_gpu_mem,
                split_type=split_type,
            )
            for cfg in self.data.explode()
        ]

        if self.out_dir is None:
            # Save in a memorably-named directory inside of
            # ELK_REPORTER_DIR/<model_name>/<dataset_name>
            ds_name = "+".join(self.data.datasets)
            root = elk_reporter_dir() / self.data.model / ds_name

            self.out_dir = memorably_named_dir(root)

        # Print the output directory in bold with escape codes
        print(f"Output directory at \033[1m{self.out_dir}\033[0m")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # save_dc_types really ought to be the default... We simply can't load
        # properly without this flag enabled.
        save(self, self.out_dir / "cfg.yaml", save_dc_types=True)

        path = self.out_dir / "fingerprints.yaml"
        with open(path, "w") as meta_f:
            yaml.dump(
                {
                    ds_name: {split: ds[split]._fingerprint for split in ds.keys()}
                    for ds_name, ds in self.datasets
                },
                meta_f,
            )

        devices = select_usable_devices(self.num_gpus, min_memory=self.min_gpu_mem)
        num_devices = len(devices)
        func: Callable[[int], LayerApplied] = partial(
            self.apply_to_layer, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)

    @abstractmethod
    def apply_to_layer(
        self, layer: int, devices: list[str], world_size: int
    ) -> LayerApplied:
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
            key = select_split(ds, split_type)

            split = ds[key].with_format("torch", device=device, dtype=torch.int16)
            labels = assert_type(Tensor, split["label"])
            hiddens = int16_to_float32(assert_type(Tensor, split[f"hidden_{layer}"]))
            if self.prompt_indices:
                hiddens = hiddens[:, self.prompt_indices]

            with split.formatted_as("torch", device=device):
                has_preds = "model_logits" in split.features
                lm_preds = split["model_logits"] if has_preds else None

            out[ds_name] = (hiddens, labels.to(hiddens.device), lm_preds)

        return out

    def concatenate(self, layers):
        """Concatenate hidden states from a previous layer."""
        for layer in range(self.concatenated_layer_offset, len(layers)):
            layers[layer] += [layers[layer][0] - self.concatenated_layer_offset]

        return layers

    def apply_to_layers(
        self,
        func: Callable[[int], LayerApplied],
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

        layers, *rest = [get_layer_indices(ds) for _, ds in self.datasets]
        assert all(x == layers for x in rest), "All datasets must have the same layers"

        if self.concatenated_layer_offset > 0:
            layers = self.concatenate(layers)

        ctx = mp.get_context("spawn")
        with ctx.Pool(num_devices) as pool:
            mapper = pool.imap_unordered if num_devices > 1 else map
            df_buffers = defaultdict(list)
            layer_outputs: list[LayerOutput] = []
            try:
                for res in tqdm(mapper(func, layers), total=len(layers)):
                    layer_outputs.extend(res.layer_outputs)
                    for k, v in res.df_dict.items():  # type: ignore
                        df_buffers[k].append(v)
            finally:
                # Make sure the CSVs are written even if we crash or get interrupted
                for name, dfs in df_buffers.items():
                    df = pd.concat(dfs).sort_values(by=["layer", PROMPT_ENSEMBLING])
                    df.round(4).to_csv(self.out_dir / f"{name}.csv", index=False)
                if self.debug:
                    save_debug_log(self.datasets, self.out_dir)

                calculate_layer_outputs(
                    layer_outputs=layer_outputs,
                    out_path=self.out_dir / "layer_ensembling.csv",
                )
