"""Main training loop."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import Serializable, field, subgroups
from sklearn.metrics import roc_auc_score

from ..extraction.extraction import Extract
from ..metrics import accuracy, to_one_hot
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

        (train_h, train_labels, _), *rest = train_dict.values()
        (n, v, k, d) = train_h.shape

        if not all(other_h.shape[2] == k for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same number of classes")

        # Can't figure out a way to make this line less ugly
        next(iter(train_dict.values()))[0].shape[-1]
        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        if isinstance(self.cfg.net, CcsReporterConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"

            reporter = CcsReporter(self.cfg.net, d, device=device)
            train_loss = reporter.fit(train_h, train_labels)

            (val_h, val_gt, _) = next(iter(val_dict.values()))
            x0, x1 = train_h.unbind(2)
            val_x0, val_x1 = val_h.unbind(2)
            pseudo_auroc = reporter.check_separability(
                train_pair=(x0, x1), val_pair=(val_x0, val_x1)
            )

        elif isinstance(self.cfg.net, EigenReporterConfig):
            # To enable training on multiple tasks with different numbers of variants,
            # we update the statistics in a streaming fashion and then fit
            reporter = EigenReporter(self.cfg.net, d, k, device=device)

            hidden_list, label_list = [], []
            for ds_name, (train_h, train_labels, _) in train_dict.items():
                hidden_list.append(train_h)
                label_list.append(train_labels)
                reporter.update(train_h)

            pseudo_auroc = None
            train_loss = reporter.fit_streaming()
            reporter.platt_scale(
                to_one_hot(
                    repeat(torch.cat(label_list), "n -> (n v)", v=v), k
                ).flatten(),
                rearrange(torch.cat(hidden_list), "n v k d -> (n v k) d"),
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

            if val_lm_preds is not None:
                val_gt_cpu = repeat(val_gt, "n -> (n v)", v=v).cpu()
                val_lm_preds = rearrange(val_lm_preds, "n v ... -> (n v) ...")
                val_lm_auroc = roc_auc_score(
                    to_one_hot(val_gt_cpu, k).long().flatten(), val_lm_preds.flatten()
                )

                val_lm_acc = accuracy(val_gt_cpu, torch.from_numpy(val_lm_preds))
            else:
                val_lm_auroc = None
                val_lm_acc = None

            row = pd.Series(
                {
                    "dataset": ds_name,
                    "layer": layer,
                    "pseudo_auroc": pseudo_auroc,
                    "train_loss": train_loss,
                    **val_result._asdict(),
                    "lm_auroc": val_lm_auroc,
                    "lm_acc": val_lm_acc,
                }
            )

            if lr_model is not None:
                row["lr_auroc"], row["lr_acc"] = evaluate_supervised(
                    lr_model, val_h, val_gt
                )

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
