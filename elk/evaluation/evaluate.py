from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing.helpers import Serializable, field
from sklearn.metrics import roc_auc_score

from elk.metrics import accuracy, to_one_hot

from ..extraction.extraction import Extract
from ..files import elk_reporter_dir
from ..run import Run
from ..training import Reporter
from ..training.supervised import evaluate_supervised
from ..utils import select_usable_devices


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
        skip_supervised: Whether to skip evaluation of the supervised classifier.
        debug: When in debug mode, a useful log file is saved to the memorably-named
            output directory. Defaults to False.
    """

    data: Extract
    source: str = field(positional=True)

    preds_out_dir: Path | None = None
    concatenated_layer_offset: int = 0
    debug: bool = False
    min_gpu_mem: int | None = None
    num_gpus: int = -1
    skip_supervised: bool = False

    def execute(self):
        transfer_dir = elk_reporter_dir() / self.source / "transfer_eval"

        for dataset in self.data.prompts.datasets:
            run = Evaluate(cfg=self, out_dir=transfer_dir / dataset)

            run.evaluate()


@dataclass
class Evaluate(Run):
    cfg: Eval

    def evaluate_reporter(
        self, layer: int, devices: list[str], world_size: int = 1
    ) -> tuple[pd.DataFrame, dict]:
        """Evaluate a single reporter on a single layer."""
        is_raw = self.cfg.data.prompts.datasets == ["raw"]
        device = self.get_device(devices, world_size)

        if is_raw:
            assert len(self.datasets) == 1
            ds = self.datasets[0]["val"]
            val_output = {"raw": self.prepare_individual(ds, device, layer)}
        else:
            val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.cfg.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        row_buf = []
        preds_buf = dict()
        for ds_name, (val_h, val_gt, val_lm_preds) in val_output.items():
            with torch.no_grad():
                ds_preds = {f"reporter_{layer}": reporter(val_h).cpu().numpy()}
            val_result = (
                reporter.score(val_gt, val_h)
                if is_raw
                else reporter.score_contrast_set(val_gt, val_h)
            )

            if val_lm_preds is not None:
                (_, v, k, _) = val_h.shape

                val_gt_cpu = repeat(val_gt, "n -> (n v)", v=v).cpu()
                val_lm_preds = rearrange(val_lm_preds, "n v ... -> (n v) ...")
                val_lm_auroc = roc_auc_score(
                    to_one_hot(val_gt_cpu, k).long().flatten(), val_lm_preds.flatten()
                )

                val_lm_acc = accuracy(val_gt_cpu, torch.from_numpy(val_lm_preds))
            else:
                val_lm_auroc = None
                val_lm_acc = None

            stats_row = pd.Series(
                {
                    "dataset": ds_name,
                    "layer": layer,
                    **val_result._asdict(),
                    "lm_auroc": val_lm_auroc,
                    "lm_acc": val_lm_acc,
                }
            )

            lr_dir = experiment_dir / "lr_models"
            if not self.cfg.skip_supervised and lr_dir.exists():
                with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
                    lr_model = torch.load(f, map_location=device).eval()

                lr_auroc_res, lr_acc = evaluate_supervised(lr_model, val_h, val_gt)
                with torch.no_grad():
                    ds_preds[f"lr_{layer}"] = lr_model(val_h).cpu().numpy().squeeze(-1)

                stats_row["lr_auroc"] = lr_auroc_res.estimate
                stats_row["lr_auroc_lower"] = lr_auroc_res.lower
                stats_row["lr_auroc_upper"] = lr_auroc_res.upper
                stats_row["lr_acc"] = lr_acc

            row_buf.append(stats_row)
            preds_buf[ds_name] = ds_preds

        return pd.DataFrame(row_buf), preds_buf

    def evaluate(self):
        """Evaluate the reporter on all layers."""
        devices = select_usable_devices(
            self.cfg.num_gpus, min_memory=self.cfg.min_gpu_mem
        )

        num_devices = len(devices)
        func: Callable[[int], tuple[pd.DataFrame, dict]] = partial(
            self.evaluate_reporter, devices=devices, world_size=num_devices
        )
        self.apply_to_layers(func=func, num_devices=num_devices)
