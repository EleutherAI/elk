from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from simple_parsing.helpers import Serializable, field

from evaluation.evaluate_log import EvalLog
from extraction.extraction import Extract
from files import elk_reporter_dir
from run import Run
from training import Reporter
from training.baseline import evaluate_baseline, load_baseline, train_baseline
from utils import select_usable_devices


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

        experiment_dir = elk_reporter_dir() / self.cfg.source

        reporter_path = (
            experiment_dir / "reporters" / f"layer_{layer}.pt"
        )
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        test_result = reporter.score(
            test_labels,
            test_x0,
            test_x1,
        )

        lr_dir = experiment_dir / "lr_models"
        if not self.cfg.skip_baseline and lr_dir.exists():
            lr_model = load_baseline(lr_dir, layer)
            lr_auroc, lr_acc = evaluate_baseline(lr_model, test_x0, test_x1, test_labels)

            print("transfer_eval", lr_auroc, lr_acc)

            # stats.lr_auroc = lr_auroc
            # stats.lr_acc = lr_acc
            # save_baseline(lr_dir, layer, lr_model)

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

import torch
from simple_parsing.helpers import Serializable, field

from evaluation.evaluate_log import EvalLog
from extraction.extraction import Extract
from files import elk_reporter_dir
from run import Run
from training import Reporter
from training.baseline import evaluate_baseline, load_baseline, train_baseline
from utils import select_usable_devices


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

        experiment_dir = elk_reporter_dir() / self.cfg.source

        reporter_path = (
            experiment_dir / "reporters" / f"layer_{layer}.pt"
        )
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        test_result = reporter.score(
            test_labels,
            test_x0,
            test_x1,
        )

        lr_dir = experiment_dir / "lr_models"
        if not self.cfg.skip_baseline and lr_dir.exists():
            lr_model = load_baseline(lr_dir, layer)
            lr_auroc, lr_acc = evaluate_baseline(lr_model, test_x0, test_x1, test_labels)

            print("transfer_eval", lr_auroc, lr_acc)

            # stats.lr_auroc = lr_auroc
            # stats.lr_acc = lr_acc
            # save_baseline(lr_dir, layer, lr_model)

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
