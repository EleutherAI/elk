from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd
import torch
from simple_parsing.helpers import Serializable, field

from ..extraction.extraction import Extract
from ..files import elk_reporter_dir, memorably_named_dir
from ..run import Run
from ..training import Reporter
from ..training.baseline import evaluate_baseline, load_baseline
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
        debug: When in debug mode, a useful log file is saved to the memorably-named
            output directory. Defaults to False.
    """

    data: Extract
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"

    debug: bool = False
    out_dir: Optional[Path] = None
    num_gpus: int = -1
    skip_baseline: bool = False
    concatenated_layer_offset: int = 0

    def execute(self):
        transfer_eval = elk_reporter_dir() / self.source / "transfer_eval"
        out_dir = memorably_named_dir(transfer_eval)

        run = Evaluate(cfg=self, out_dir=out_dir)
        run.evaluate()


@dataclass
class Evaluate(Run):
    cfg: Eval

    def evaluate_reporter(
        self, layer: int, devices: list[str], world_size: int = 1
    ) -> pd.Series:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)

        _, test_h, _, test_labels, _ = self.prepare_data(
            device,
            layer,
        )

        experiment_dir = elk_reporter_dir() / self.cfg.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        test_result = reporter.score(test_labels, test_h)
        stats_row = pd.Series(
            {
                "layer": layer,
                **test_result._asdict(),
            }
        )

        lr_dir = experiment_dir / "lr_models"
        if not self.cfg.skip_baseline and lr_dir.exists():
            lr_model = load_baseline(lr_dir, layer)
            lr_model.eval()
            lr_auroc, lr_acc = evaluate_baseline(
                lr_model.cuda(), test_h.cuda(), test_labels
            )

            stats_row["lr_auroc"] = lr_auroc
            stats_row["lr_acc"] = lr_acc

        return stats_row

    def evaluate(self):
        """Evaluate the reporter on all layers."""
        devices = select_usable_devices(
            self.cfg.num_gpus, min_memory=self.cfg.data.min_gpu_mem
        )

        num_devices = len(devices)
        func: Callable[[int], pd.Series] = partial(
            self.evaluate_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)
