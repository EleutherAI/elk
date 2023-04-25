from dataclasses import dataclass

import pandas as pd
import torch
from simple_parsing.helpers import field

from ..files import elk_reporter_dir
from ..metrics import evaluate_preds
from ..run import Run
from ..training import Reporter


@dataclass
class Eval(Run):
    """Full specification of a reporter evaluation run."""

    source: str = field(default="", positional=True)
    skip_supervised: bool = False

    def __post_init__(self):
        assert self.source, "Must specify a source experiment."

        transfer_dir = elk_reporter_dir() / self.source / "transfer_eval"
        self.out_dir = transfer_dir / "+".join(self.data.prompts.datasets)

    def apply_to_layer(
        self, layer: int, devices: list[str], world_size: int
    ) -> pd.DataFrame:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        row_buf = []
        for ds_name, (val_h, val_gt, _) in val_output.items():
            val_result = evaluate_preds(val_gt, reporter(val_h))

            stats_row = {
                "dataset": ds_name,
                "layer": layer,
                **val_result.to_dict(),
            }

            lr_dir = experiment_dir / "lr_models"
            if not self.skip_supervised and lr_dir.exists():
                with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
                    lr_model = torch.load(f, map_location=device).eval()

                lr_result = evaluate_preds(val_gt, lr_model(val_h))
                stats_row.update(lr_result.to_dict(prefix="lr_"))

            row_buf.append(stats_row)

        return pd.DataFrame.from_records(row_buf)
