"""Main training loop."""

import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from simple_parsing import Serializable, field, subgroups
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor

from ..extraction.extraction import Extract
from ..run import Run
from ..training.baseline import evaluate_baseline, train_baseline
from ..utils import select_usable_devices
from ..utils.typing import assert_type
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .reporter import OptimConfig, Reporter, ReporterConfig


@dataclass
class Elicit(Serializable):
    """Full specification of a reporter training run.

    Args:
        data: Config specifying hidden states on which the reporter will be trained.
        net: Config for building the reporter network.
        optim: Config for the `.fit()` loop.
        num_gpus: The number of GPUs to use. Defaults to -1, which means
            "use all available GPUs".
        normalization: The normalization method to use. Defaults to "meanonly". See
            `elk.training.preprocessing.normalize()` for details.
        skip_baseline: Whether to skip training the baseline classifier. Defaults to
            False.
        debug: When in debug mode, a useful log file is saved to the memorably-named
            output directory. Defaults to False.
    """

    data: Extract
    net: ReporterConfig = subgroups(
        {"ccs": CcsReporterConfig, "eigen": EigenReporterConfig}, default="eigen"
    )
    optim: OptimConfig = field(default_factory=OptimConfig)

    num_gpus: int = -1
    min_gpu_mem: int | None = None
    skip_baseline: bool = False
    concatenated_layer_offset: int = 0
    # if nonzero, appends the hidden states of layer concatenated_layer_offset before
    debug: bool = False
    out_dir: Optional[Path] = None

    def execute(self):
        train_run = Train(cfg=self, out_dir=self.out_dir)
        train_run.train()


@dataclass
class Train(Run):
    cfg: Elicit

    def create_models_dir(self, out_dir: Path):
        lr_dir = None
        lr_dir = out_dir / "lr_models"
        reporter_dir = out_dir / "reporters"

        lr_dir.mkdir(parents=True, exist_ok=True)
        reporter_dir.mkdir(parents=True, exist_ok=True)

        return reporter_dir, lr_dir

    def train_reporter(
        self,
        layer: int,
        devices: list[str],
        world_size: int = 1,
    ) -> pd.DataFrame:
        """Train a single reporter on a single layer."""
        self.make_reproducible(seed=self.cfg.net.seed + layer)
        device = self.get_device(devices, world_size)

        x0, x1, train_gt, val_output = self.prepare_data(device, layer)
        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        # pseudo_auroc = self.get_pseudo_auroc(layer, x0, x1, val_x0, val_x1)

        if isinstance(self.cfg.net, CcsReporterConfig):
            reporter = CcsReporter(x0.shape[-1], self.cfg.net, device=device)
        elif isinstance(self.cfg.net, EigenReporterConfig):
            reporter = EigenReporter(x0.shape[-1], self.cfg.net, device=device)
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.cfg.net)}")

        # Fit reporter
        train_loss = reporter.fit(x0, x1, train_gt)
        with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(reporter, file)

        # Fit baseline logistic regression model
        lr_model = train_baseline(x0, x1, train_gt, device=device)
        with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(lr_model, file)

        row_buf = []
        for ds_name, (val_x0, val_x1, val_gt, val_lm_preds) in val_output.items():
            val_result = reporter.score(
                val_gt,
                val_x0,
                val_x1,
            )

            if val_lm_preds is not None:
                val_gt_cpu = (
                    val_gt.repeat_interleave(val_lm_preds.shape[1]).float().cpu()
                )
                val_lm_auroc = float(roc_auc_score(val_gt_cpu, val_lm_preds.flatten()))
                val_lm_acc = float(
                    accuracy_score(val_gt_cpu, val_lm_preds.flatten() > 0.5)
                )
            else:
                val_lm_auroc = None
                val_lm_acc = None

            row = pd.Series(
                {
                    "dataset": ds_name,
                    "layer": layer,
                    # "pseudo_auroc": pseudo_auroc,
                    "train_loss": train_loss,
                    **val_result._asdict(),
                    "lm_auroc": val_lm_auroc,
                    "lm_acc": val_lm_acc,
                }
            )

            lr_auroc, lr_acc = evaluate_baseline(lr_model, val_x0, val_x1, val_gt)

            row["lr_auroc"] = lr_auroc
            row["lr_acc"] = lr_acc
            row_buf.append(row)

        return pd.DataFrame(row_buf)

    def get_pseudo_auroc(
        self, layer: int, x0: Tensor, x1: Tensor, val_x0: Tensor, val_x1: Tensor
    ):
        """Check the separability of the pseudo-labels at a given layer."""

        with torch.no_grad():
            pseudo_auroc = Reporter.check_separability(
                train_pair=(x0, x1), val_pair=(val_x0, val_x1)
            )
            if pseudo_auroc > 0.6:
                warnings.warn(
                    f"The pseudo-labels at layer {layer} are linearly separable with "
                    f"an AUROC of {pseudo_auroc:.3f}. This may indicate that the "
                    f"algorithm will not converge to a good solution."
                )

        return pseudo_auroc

    def train(self):
        """Train a reporter on each layer of the network."""
        devices = select_usable_devices(self.cfg.num_gpus)
        num_devices = len(devices)
        func: Callable[[int], pd.DataFrame] = partial(
            self.train_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)
