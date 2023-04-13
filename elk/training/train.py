"""Main training loop."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import Serializable, field, subgroups
from sklearn.metrics import accuracy_score, roc_auc_score

from ..extraction.extraction import Extract
from ..run import Run
from ..training.supervised import evaluate_supervised, train_supervised
from ..utils import select_usable_devices
from ..utils.typing import assert_type
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .classifier import Classifier
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
        skip_baseline: Whether to skip training the supervised classifier. Defaults to
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

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        # Can't figure out a way to make this line less ugly
        hidden_size = next(iter(train_dict.values()))[0].shape[-1]
        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        pseudo_clf = self.get_pseudo_classifier(train_dict, device)

        if isinstance(self.cfg.net, CcsReporterConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"

            reporter = CcsReporter(hidden_size, self.cfg.net, device=device)
            (x0, x1, labels, _) = next(iter(train_dict.values()))
            train_loss = reporter.fit(x0, x1, labels)

        elif isinstance(self.cfg.net, EigenReporterConfig):
            # To enable training on multiple tasks with different numbers of variants,
            # we update the statistics in a streaming fashion and then fit
            reporter = EigenReporter(hidden_size, self.cfg.net, device=device)
            for ds_name, (x0, x1, labels, _) in train_dict.items():
                reporter.update(x0, x1)

            train_loss = reporter.fit_streaming()
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.cfg.net)}")

        # Save reporter checkpoint to disk
        with open(reporter_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(reporter, file)

        # Fit supervised logistic regression model
        lr_model = train_supervised(train_dict, device=device)
        with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(lr_model, file)

        row_buf = []
        for ds_name, (val_x0, val_x1, val_gt, val_lm_preds) in val_dict.items():
            val_result = reporter.score(
                val_gt,
                val_x0,
                val_x1,
            )
            with torch.no_grad():
                (n, v, d) = val_x0.shape

                pseudo_preds = pseudo_clf(
                    # b v d -> (b v) d
                    torch.cat([val_x0, val_x1]).flatten(0, 1)
                )
                pseudo_labels = torch.cat(
                    [
                        val_x0.new_zeros(n),
                        val_x0.new_ones(n),
                    ]
                )
                pseudo_labels = repeat(pseudo_labels, "n -> (n v)", v=v)
                pseudo_auroc = float(
                    roc_auc_score(pseudo_labels.cpu(), pseudo_preds.cpu())
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
                    "pseudo_auroc": pseudo_auroc,
                    "train_loss": train_loss,
                    **val_result._asdict(),
                    "lm_auroc": val_lm_auroc,
                    "lm_acc": val_lm_acc,
                }
            )

            lr_auroc, lr_acc = evaluate_supervised(lr_model, val_x0, val_x1, val_gt)

            row["lr_auroc"] = lr_auroc
            row["lr_acc"] = lr_acc
            row_buf.append(row)

        return pd.DataFrame(row_buf)

    def get_pseudo_classifier(self, data: dict[str, tuple], device: str) -> Classifier:
        """Check the separability of the pseudo-labels at a given layer."""

        x0s, x1s = [], []
        for x0, x1, _, _ in data.values():
            x0s.append(rearrange(x0, "n v d -> (n v) d"))
            x1s.append(rearrange(x1, "n v d -> (n v) d"))

        # Simple de-meaning normalization
        X0 = torch.cat(x0s)
        X1 = torch.cat(x1s)
        X0 -= X0.mean(dim=0)
        X1 -= X1.mean(dim=0)

        X = torch.cat([X0, X1])
        Y = torch.cat([X0.new_zeros(X0.shape[0]), X0.new_ones(X1.shape[0])])

        pseudo_clf = Classifier(X.shape[-1], device=device)
        pseudo_clf.fit(X, Y)
        return pseudo_clf

    def train(self):
        """Train a reporter on each layer of the network."""
        devices = select_usable_devices(self.cfg.num_gpus)
        num_devices = len(devices)
        func: Callable[[int], pd.DataFrame] = partial(
            self.train_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)
