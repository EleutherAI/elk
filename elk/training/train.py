"""Main training loop."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import Serializable, field, subgroups

from ..extraction.extraction import Extract
from ..metrics import accuracy, roc_auc_ci, to_one_hot
from ..run import Run
from ..training.supervised import evaluate_supervised, train_supervised
from ..utils import select_usable_devices
from ..utils.typing import assert_type
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .reporter import OptimConfig, ReporterConfig


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
        supervised: Whether to train a supervised classifier, and if so, whether to
            use cross-validation. Defaults to "single", which means to train a single
            classifier on the training data. "cv" means to use cross-validation.
        debug: When in debug mode, a useful log file is saved to the memorably-named
            output directory. Defaults to False.
    """

    data: Extract
    net: ReporterConfig = subgroups(
        {"ccs": CcsReporterConfig, "eigen": EigenReporterConfig}, default="eigen"
    )
    optim: OptimConfig = field(default_factory=OptimConfig)

    concatenated_layer_offset: int = 0
    debug: bool = False
    min_gpu_mem: int | None = None
    num_gpus: int = -1
    out_dir: Path | None = None
    supervised: Literal["none", "single", "cv"] = "single"

    def execute(self):
        Train(cfg=self, out_dir=self.out_dir).train()


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

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        (first_train_h, train_labels, _), *rest = train_dict.values()
        d = first_train_h.shape[-1]
        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        if isinstance(self.cfg.net, CcsReporterConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"

            reporter = CcsReporter(self.cfg.net, d, device=device)
            train_loss = reporter.fit(first_train_h, train_labels)

            (val_h, val_gt, _) = next(iter(val_dict.values()))
            x0, x1 = first_train_h.unbind(2)
            val_x0, val_x1 = val_h.unbind(2)
            pseudo_auroc = reporter.check_separability(
                train_pair=(x0, x1), val_pair=(val_x0, val_x1)
            )

        elif isinstance(self.cfg.net, EigenReporterConfig):
            # We set num_classes to None to enable training on datasets with different
            # numbers of classes. Under the hood, this causes the covariance statistics
            # to be simply averaged across all batches passed to update().
            reporter = EigenReporter(self.cfg.net, d, num_classes=None, device=device)

            hidden_list, label_list = [], []
            for ds_name, (train_h, train_labels, _) in train_dict.items():
                (_, v, k, _) = train_h.shape

                # Datasets can have different numbers of variants and different numbers
                # of classes, so we need to flatten them here before concatenating
                hidden_list.append(rearrange(train_h, "n v k d -> (n v k) d"))
                label_list.append(
                    to_one_hot(repeat(train_labels, "n -> (n v)", v=v), k).flatten()
                )
                reporter.update(train_h)

            pseudo_auroc = None
            train_loss = reporter.fit_streaming()
            reporter.platt_scale(
                torch.cat(label_list),
                torch.cat(hidden_list),
            )
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.cfg.net)}")

        # Save reporter checkpoint to disk
        with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(reporter, file)

        # Fit supervised logistic regression model
        if self.cfg.supervised != "none":
            lr_model = train_supervised(
                train_dict, device=device, cv=self.cfg.supervised == "cv"
            )
            with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_model, file)
        else:
            lr_model = None

        row_buf = []
        for ds_name, (val_h, val_gt, val_lm_preds) in val_dict.items():
            val_result = reporter.score(val_gt, val_h)
            row = pd.Series(
                {
                    "dataset": ds_name,
                    "layer": layer,
                    "pseudo_auroc": pseudo_auroc,
                    "train_loss": train_loss,
                    **val_result._asdict(),
                }
            )

            if val_lm_preds is not None:
                (_, v, k, _) = val_h.shape

                val_gt_rep = repeat(val_gt, "n -> (n v)", v=v)
                val_lm_preds = rearrange(val_lm_preds, "n v ... -> (n v) ...")
                val_lm_auroc_res = roc_auc_ci(
                    to_one_hot(val_gt_rep, k).long().flatten(), val_lm_preds.flatten()
                )
                row["lm_auroc"] = val_lm_auroc_res.estimate
                row["lm_auroc_lower"] = val_lm_auroc_res.lower
                row["lm_auroc_upper"] = val_lm_auroc_res.upper
                row["lm_acc"] = accuracy(val_gt_rep, val_lm_preds)

            if lr_model is not None:
                lr_auroc_res, row["lr_acc"] = evaluate_supervised(
                    lr_model, val_h, val_gt
                )
                row["lr_auroc"] = lr_auroc_res.estimate
                row["lr_auroc_lower"] = lr_auroc_res.lower
                row["lr_auroc_upper"] = lr_auroc_res.upper

            row_buf.append(row)

        return pd.DataFrame(row_buf)

    def train(self):
        """Train a reporter on each layer of the network."""
        devices = select_usable_devices(self.cfg.num_gpus)
        num_devices = len(devices)
        func: Callable[[int], pd.DataFrame] = partial(
            self.train_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)
