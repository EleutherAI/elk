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
from .utils import (
    Color,
    assert_type,
    get_layer_indices,
    int16_to_float32,
    select_split,
    select_usable_devices,
)


@dataclass
class LayerData:
    hiddens: Tensor
    labels: Tensor
    lm_log_odds: Tensor | None
    texts: list[list[str]]  # (n, v)
    row_ids: list[int]  # (n,)
    variant_ids: list[list[str]]  # (n, v)


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

    save_logprobs: bool = field(default=False, to_dict=False)
    """ saves logprobs.pt containing
        {<dsname>: {"row_ids": [n,], "variant_ids": [n, v],
            "labels": [n,], "texts": [n, v],
            "lm": {"none": [n, v], "full": [n,]},
            "lr": {<layer>: {<inlp_iter>: {"none": [n, v], "full": [n,]}}}
        }}
    """

    concatenated_layer_offset: int = 0
    debug: bool = False
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

        devices = select_usable_devices(self.num_gpus)
        num_devices = len(devices)
        func: Callable[[int], tuple[dict[str, pd.DataFrame], dict]] = partial(
            self.apply_to_layer, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)

    @abstractmethod
    def apply_to_layer(
        self, layer: int, devices: list[str], world_size: int
    ) -> tuple[dict[str, pd.DataFrame], dict]:
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
    ) -> dict[str, LayerData]:
        """Prepare data for the specified layer and split type."""
        out = {}

        for ds_name, ds in self.datasets:
            key = select_split(ds, split_type)

            split = ds[key].with_format("torch", device=device, dtype=torch.int16)
            labels = assert_type(Tensor, split["label"])
            # hiddens shape: (num_examples, num_variants, hidden_d)
            hiddens = int16_to_float32(assert_type(Tensor, split[f"hidden_{layer}"]))
            if self.prompt_indices:
                hiddens = hiddens[:, self.prompt_indices]

            if "lm_log_odds" in split.column_names:
                with split.formatted_as("torch", device=device):
                    lm_preds = assert_type(Tensor, split["lm_log_odds"])
            else:
                lm_preds = None

            out[ds_name] = LayerData(
                hiddens=hiddens,
                labels=labels,
                lm_log_odds=lm_preds,
                texts=split["texts"],
                row_ids=split["row_id"],
                variant_ids=split["variant_ids"],
            )

        return out

    def concatenate(self, layers):
        """Concatenate hidden states from a previous layer."""
        for layer in range(self.concatenated_layer_offset, len(layers)):
            layers[layer] += [layers[layer][0] - self.concatenated_layer_offset]

        return layers

    def apply_to_layers(
        self,
        func: Callable[[int], tuple[dict[str, pd.DataFrame], dict]],
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
            logprobs_dicts = defaultdict(dict)

            try:
                for layer, (df_dict, logprobs_dict) in tqdm(
                    zip(layers, mapper(func, layers)), total=len(layers)
                ):
                    for k, v in df_dict.items():
                        df_buffers[k].append(v)
                    for k, v in logprobs_dict.items():
                        logprobs_dicts[k][layer] = logprobs_dict[k]
            finally:
                # Make sure the CSVs are written even if we crash or get interrupted
                for name, dfs in df_buffers.items():
                    df = pd.concat(dfs).sort_values(by=["layer", "ensembling"])
                    df.round(4).to_csv(self.out_dir / f"{name}.csv", index=False)
                if self.debug:
                    save_debug_log(self.datasets, self.out_dir)
                if self.save_logprobs:
                    save_dict = defaultdict(dict)
                    for ds_name, logprobs_dict in logprobs_dicts.items():
                        save_dict[ds_name]["texts"] = logprobs_dict[layers[0]]["texts"]
                        save_dict[ds_name]["labels"] = logprobs_dict[layers[0]][
                            "labels"
                        ]
                        save_dict[ds_name]["lm"] = logprobs_dict[layers[0]]["lm"]
                        save_dict[ds_name]["reporter"] = dict()
                        save_dict[ds_name]["lr"] = dict()
                        for layer, logprobs_dict_by_mode in logprobs_dict.items():
                            save_dict[ds_name]["reporter"][
                                layer
                            ] = logprobs_dict_by_mode["reporter"]
                            save_dict[ds_name]["lr"][layer] = logprobs_dict_by_mode[
                                "lr"
                            ]
                    torch.save(save_dict, self.out_dir / "logprobs.pt")
