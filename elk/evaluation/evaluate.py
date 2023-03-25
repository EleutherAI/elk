from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from simple_parsing import Serializable, field

from datasets import Split
from elk.extraction.extraction import Extract
from elk.files import elk_reporter_dir
from elk.run import Run


@dataclass
class Eval(Serializable):
    """
    Full specification of a reporter evaluation run.

    Args:
        data: Config specifying hidden states on which the reporter will be evaluated.
        source: The name of the source run directory which contains the reporters directory.
        normalization: The normalization method to use. Defaults to "meanonly". See
            `elk.training.preprocessing.normalize()` for details.
        max_gpus: The maximum number of GPUs to use. Defaults to -1, which means
            "use all available GPUs".
        debug: When in debug mode, a useful log file is saved to the memorably-named
            output directory. Defaults to False.
    """

    data: Extract
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    max_gpus: int = -1
    debug: bool = False
    out_dir: Optional[Path] = None

    def execute(self):
        transfer_eval = elk_reporter_dir() / self.source / "transfer_eval"
        cols = ["layer", "loss", "acc", "cal_acc", "auroc"]

        run = Evaluate(cfg=self, eval_headers=cols, out_dir=transfer_eval)
        run.evaluate()


@dataclass
class Evaluate(Run):
    cfg: Eval

    def evaluate_reporter(self, layer: int, devices: list[str], world_size: int):
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)

        _, _, test_x0, test_x1, _, test_labels = self.prepare_data(
            device,
            layer,
            priorities={Split.TRAIN: 0, Split.VALIDATION: 1, Split.TEST: 2},
        )

        reporter_path = (
            elk_reporter_dir() / self.cfg.source / "reporters" / f"layer_{layer}.pt"
        )
        reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        test_result = reporter.score(
            test_labels,
            test_x0,
            test_x1,
        )

        stats = [layer, *test_result]
        return stats

    def evaluate(self):
        """Evaluate the reporter on all layers."""

        self.apply_to_layers(func=self.evaluate_reporter)
