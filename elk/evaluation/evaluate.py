from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from simple_parsing.helpers import Serializable, field

from elk.evaluation.evaluate_log import EvalLog
from elk.extraction.extraction import Extract
from elk.run import Run
from elk.training import Reporter

from ..files import elk_reporter_dir
from ..utils import (
    select_usable_devices,
)


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
    out_dir_suffix: Optional[
        Path
    ] = None  # custom name for subdir in transfer_eval folder
    num_gpus: int = -1

    concatenated_layer_offset: int = 0

    def execute(self):
        if self.out_dir_suffix is None:
            self.out_dir_suffix = "-".join(self.data.prompts.datasets).replace(" ", "_")

        transfer_eval = (
            elk_reporter_dir() / self.source / "transfer_eval" / self.out_dir_suffix
        )

        run = Evaluate(cfg=self, out_dir=transfer_eval)
        run.evaluate()


@dataclass
class Evaluate(Run):
    cfg: Eval

    def evaluate_reporter(
        self, layer: int, devices: list[str], world_size: int = 1
    ) -> EvalLog:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)

        _, _, test_x0, test_x1, _, test_labels = self.prepare_data(
            device,
            layer,
        )

        reporter_path = (
            elk_reporter_dir() / self.cfg.source / "reporters" / f"layer_{layer}.pt"
        )
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        test_result = reporter.score(
            test_labels,
            test_x0,
            test_x1,
        )

        return EvalLog(
            layer=layer,
            eval_result=test_result,
        )

    def evaluate(self):
        """Evaluate the reporter on all layers."""
        devices = select_usable_devices(
            self.cfg.num_gpus, min_memory=self.cfg.data.min_gpu_mem
        )

        num_devices = len(devices)
        func: Callable[[int], EvalLog] = partial(
            self.evaluate_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(
            func=func,
            num_devices=num_devices,
            to_csv_line=lambda item: item.to_csv_line(),
            csv_columns=EvalLog.csv_columns(),
        )
