from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from simple_parsing.helpers import Serializable, field

from ..extraction.extraction import Extract
from ..files import elk_reporter_dir
from ..run import Run
from ..training import Reporter
from ..training.supervised import evaluate_supervised
from ..utils import select_usable_devices


@dataclass
class Eval(Serializable):
    """
    Full specification of a reporter evaluation run.

    Args:
        data: Config specifying hidden states on which the reporter will be evaluated.
        source: The name of the source run directory
            which contains the reporters directory.
        normalization: The normalization method to use. Defaults to "meanonly". See
            `elk.training.preprocessing.normalize()` for details.
        num_gpus: The number of GPUs to use. Defaults to -1, which means
            "use all available GPUs".
        skip_supervised: Whether to skip evaluation of the supervised classifier.
        debug: When in debug mode, a useful log file is saved to the memorably-named
            output directory. Defaults to False.
    """

    data: Extract
    source: str = field(positional=True)

    concatenated_layer_offset: int = 0
    debug: bool = False
    min_gpu_mem: int | None = None
    num_gpus: int = -1
    out_dir: Path | None = None
    skip_supervised: bool = False

    def execute(self):
        transfer_dir = elk_reporter_dir() / self.source / "transfer_eval"

        for dataset in self.data.prompts.datasets:
            run = Evaluate(cfg=self, out_dir=transfer_dir / dataset)
            run.evaluate()


@dataclass
class Evaluate(Run):
    cfg: Eval

    def evaluate_reporter(
        self, layer: int, devices: list[str], world_size: int = 1
    ) -> pd.DataFrame:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.cfg.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        row_buf = []
        for ds_name, (val_x0, val_x1, val_gt, _) in val_output.items():
            val_result = reporter.score(
                val_gt,
                val_x0,
                val_x1,
            )

            stats_row = pd.Series(
                {
                    "dataset": ds_name,
                    "layer": layer,
                    **val_result._asdict(),
                }
            )

            lr_dir = experiment_dir / "lr_models"
            if not self.cfg.skip_supervised and lr_dir.exists():
                with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
                    lr_model = torch.load(f, map_location=device).eval()

                lr_auroc, lr_acc = evaluate_supervised(lr_model, val_x0, val_x1, val_gt)

                stats_row["lr_auroc"] = lr_auroc
                stats_row["lr_acc"] = lr_acc

            row_buf.append(stats_row)

        return pd.DataFrame(row_buf)

    def evaluate(self):
        """Evaluate the reporter on all layers."""
        devices = select_usable_devices(
            self.cfg.num_gpus, min_memory=self.cfg.min_gpu_mem
        )

        num_devices = len(devices)
        func: Callable[[int], pd.DataFrame] = partial(
            self.evaluate_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)
