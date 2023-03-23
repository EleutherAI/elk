"""Main training loop."""

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from simple_parsing import Serializable, field, subgroups
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor

from datasets import DatasetDict
from elk.extraction.extraction import Extract
from elk.run import Run
from elk.utils.typing import assert_type

from .ccs_reporter import CcsReporter, CcsReporterConfig
from .classifier import Classifier
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .reporter import OptimConfig, Reporter, ReporterConfig
import yaml


@dataclass
class Elicit(Serializable):
    """Full specification of a reporter training run.

    Args:
        data: Config specifying hidden states on which the reporter will be trained.
        net: Config for building the reporter network.
        optim: Config for the `.fit()` loop.
        max_gpus: The maximum number of GPUs to use. Defaults to -1, which means
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
    max_gpus: int = -1
    normalization: Literal["legacy", "none", "elementwise", "meanonly"] = "meanonly"
    skip_baseline: bool = False
    debug: bool = False
    out_dir: Optional[Path] = None

    def execute(self):
        cols = [
            "layer",
            "pseudo_auroc",
            "train_loss",
            "acc",
            "cal_acc",
            "auroc",
            "ece",
        ]
        if not self.skip_baseline:
            cols += ["lr_auroc", "lr_acc"]
        train_run = TrainRun(cfg=self, eval_headers=cols, out_dir=self.out_dir)
        train_run.train()


@dataclass
class TrainRun(Run):
    cfg: Elicit

    def train_baseline(
        self,
        x0: Tensor,
        x1: Tensor,
        val_x0: Tensor,
        val_x1: Tensor,
        train_labels: Tensor,
        val_labels: Tensor,
        device: str,
    ):
        # repeat_interleave makes `num_variants` copies of each label, all within a
        # single dimension of size `num_variants * 2 * n`, such that the labels align
        # with X.view(-1, X.shape[-1])
        train_labels_aug = torch.cat(
            [train_labels, 1 - train_labels]
        ).repeat_interleave(x0.shape[1])
        val_labels_aug = (
            torch.cat([val_labels, 1 - val_labels]).repeat_interleave(x0.shape[1])
        ).cpu()

        X = torch.cat([x0, x1]).squeeze()
        d = X.shape[-1]
        lr_model = Classifier(d, device=device)
        lr_model.fit_cv(X.view(-1, d), train_labels_aug)

        X_val = torch.cat([val_x0, val_x1]).view(-1, d)
        with torch.no_grad():
            lr_preds = lr_model(X_val).sigmoid().cpu()

        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)

        return lr_model, lr_auroc, lr_acc

    def create_models_dir(self, out_dir: Path):
        lr_dir = None
        lr_dir = out_dir / "lr_models"
        reporter_dir = out_dir / "reporters"

        lr_dir.mkdir(parents=True, exist_ok=True)
        reporter_dir.mkdir(parents=True, exist_ok=True)

        return reporter_dir, lr_dir

    def save_baseline(self, lr_dir: Path, layer: int, lr_model: Classifier):
        with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
            pickle.dump(lr_model, file)

    def train_reporter(
        self,
        layer: int,
        devices: list[str],
        world_size: int = 1,
    ):
        """Train a single reporter on a single layer."""
        self.make_reproducible(seed=self.cfg.net.seed + layer)

        device = self.get_device(devices, world_size)

        x0, x1, val_x0, val_x1, train_labels, val_labels = self.prepare_data(
            device, layer
        )
        pseudo_auroc = self.get_pseudo_auroc(layer, x0, x1, val_x0, val_x1)

        if isinstance(self.cfg.net, CcsReporterConfig):
            reporter = CcsReporter(x0.shape[-1], self.cfg.net, device=device)
        elif isinstance(self.cfg.net, EigenReporterConfig):
            reporter = EigenReporter(x0.shape[-1], self.cfg.net, device=device)
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.cfg.net)}")

        train_loss = reporter.fit(x0, x1, train_labels)
        val_result = reporter.score(
            val_labels,
            val_x0,
            val_x1,
        )

        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        stats = [layer, pseudo_auroc, train_loss, *val_result]

        if not self.cfg.skip_baseline:
            lr_model, lr_auroc, lr_acc = self.train_baseline(
                x0,
                x1,
                val_x0,
                val_x1,
                train_labels,
                val_labels,
                device,
            )
            stats += [lr_auroc, lr_acc]
            self.save_baseline(lr_dir, layer, lr_model)

        with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(reporter, file)

        return stats

    def get_pseudo_auroc(
        self, layer: int, x0: Tensor, x1: Tensor, val_x0: Tensor, val_x1: Tensor
    ):
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
        self.apply_to_layers(func=self.train_reporter)