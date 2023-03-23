from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from simple_parsing import Serializable, field
from torch import Tensor

from datasets import DatasetDict, Split
from elk.extraction.extraction import Extract
from elk.files import create_output_directory, elk_reporter_dir
from elk.run import Run
from elk.training.preprocessing import normalize
from elk.utils.data_utils import select_train_val_splits
from elk.utils.typing import upcast_hiddens


@dataclass
class Eval(Serializable):
    data: Extract
    source: str = field(positional=True)
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    max_gpus: int = -1

    out_dir: Optional[Path] = None

    def execute(self):
        evaluate_run = EvaluateRun(cfg=self) 
        evaluate_run.evaluate_reporters()


@dataclass
class EvaluateRun(Run):
    def __post_init__(self):
        transfer_eval = elk_reporter_dir() / self.cfg.source / "transfer_eval"
        self.cfg.out_dir = create_output_directory(self.cfg.out_dir, default_root_dir=transfer_eval) 

    def evaluate_reporter(
        self,
        dataset: DatasetDict,
        out_dir: Path,
        layer: int,
        devices: list[str],
        world_size: int = 1,
    ):
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)

        _, _, test_x0, test_x1, _, test_labels = self.prepare_data(dataset, 
                                                                    device, 
                                                                    layer, 
                                                                    priorities= {Split.TRAIN: 0, Split.VALIDATION: 1, Split.TEST: 2}) 
        
        reporter_path = elk_reporter_dir() / self.cfg.source / "reporters" / f"layer_{layer}.pt"
        reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        test_result = reporter.score(
            test_labels,
            test_x0, 
            test_x1,
        )

        stats = [layer, *test_result]
        return stats


    def evaluate_reporters(self):
        cols=["layer", "loss", "acc", "cal_acc", "auroc"]
        self.run(func=self.evaluate_reporter, cols=cols)