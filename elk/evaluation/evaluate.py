from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
from simple_parsing import Serializable, field

from datasets import Split
from elk.extraction.extraction import Extract
from elk.files import elk_reporter_dir
from elk.run import Run


@dataclass
class Eval(Serializable):
    data: Extract
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    max_gpus: int = -1
    debug: bool = False
    out_dir: Optional[Path] = None

    def execute(self):
        transfer_eval = elk_reporter_dir() / self.source / "transfer_eval"
        cols = ["layer", "loss", "acc", "cal_acc", "auroc"]

        run = EvaluateRun(cfg=self, eval_headers=cols, out_dir=transfer_eval)
        run.evaluate()


@dataclass
class EvaluateRun(Run):
    cfg: Eval

    def evaluate_reporter(self, layer: int, devices: List[str], world_size: int):
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
        self.apply_to_layers(func=self.evaluate_reporter)
