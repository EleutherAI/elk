import os
from pathlib import Path
from dataclasses import dataclass
from typing import cast, Optional, Literal
import torch
from simple_parsing.helpers import Serializable, field
from torch import Tensor

from datasets import DatasetDict, Split
from elk.files import create_output_directory, elk_reporter_dir, save_config
from elk.training.preprocessing import normalize
from elk.utils.data_utils import get_layers, select_train_val_splits
from elk.utils.typing import upcast_hiddens
from elk.extraction.extraction import extract
from elk.parallelization import run_on_layers

from ..extraction import ExtractionConfig


@dataclass
class Eval(Serializable):
    target: ExtractionConfig
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    max_gpus: int = -1

    def execute(self):
        evaluate_reporters(cfg=self)


def evaluate_reporter(
    cfg: Eval,
    dataset: DatasetDict,
    layer: int,
    devices: list[str],
    world_size: int = 1,
):
    """Evaluate a single reporter on a single layer."""
    rank = os.getpid() % world_size
    device = devices[rank]

    with dataset.formatted_as("torch", device=device, dtype=torch.int16):
        train_split, test_split= select_train_val_splits(dataset, priorities= {
            Split.TRAIN: 0,
            Split.TEST: 1,
            Split.VALIDATION: 2
        })
        train, test = dataset[train_split], dataset[test_split]
        test_labels = cast(Tensor, test["label"])

        _, test_h = normalize(
            upcast_hiddens(train[f"hidden_{layer}"]),  # type: ignore
            upcast_hiddens(test[f"hidden_{layer}"]), # type: ignore
            cfg.normalization
        )

    reporter_path = elk_reporter_dir() / cfg.source / "reporters" / f"layer_{layer}.pt"
    reporter = torch.load(reporter_path, map_location=device)
    reporter.eval()

    test_x0, test_x1 = test_h.unbind(dim=-2)

    test_result = reporter.score(
        test_labels,
        test_x0, 
        test_x1,
    )

    stats = [layer, *test_result]
    return stats


def evaluate_reporters(cfg: Eval, out_dir: Optional[Path] = None):
    ds = extract(cfg.target, max_gpus=cfg.max_gpus)

    layers = get_layers(ds)

    transfer_eval = elk_reporter_dir() / cfg.source / "transfer_eval"
    out_dir = create_output_directory(out_dir, default_root_dir=transfer_eval)
    
    save_config(cfg, out_dir)

    run_on_layers(
        func=evaluate_reporter,
        cols=["layer", "loss", "acc", "cal_acc", "auroc"],
        out_dir=out_dir,
        cfg=cfg,
        ds=ds,
        layers=layers
    )