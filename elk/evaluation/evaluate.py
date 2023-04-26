from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch
from simple_parsing.helpers import field

from ..files import elk_reporter_dir, transfer_eval_directory
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

        # Set the output directory to the transfer directory if it's not specified
        self.out_dir = (
            transfer_eval_directory(self.source)
            if self.out_dir is None
            else self.out_dir
        )

    def apply_to_layer(
        self, layer: int, devices: list[str], world_size: int
    ) -> dict[str, pd.DataFrame]:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter: Reporter = torch.load(reporter_path, map_location=device)
        reporter.eval()

        row_bufs = defaultdict(list)
        for ds_name, (val_h, val_gt, _) in val_output.items():
            meta = {"dataset": ds_name, "layer": layer}

            val_credences = reporter(val_h)
            for mode in ("none", "partial", "full"):
                row_bufs["eval"].append(
                    {
                        **meta,
                        "ensembling": mode,
                        **evaluate_preds(val_gt, val_credences, mode).to_dict(),
                    }
                )

                lr_dir = experiment_dir / "lr_models"
                if not self.skip_supervised and lr_dir.exists():
                    with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
                        lr_models = torch.load(f, map_location=device)
                        if not isinstance(lr_models, list):  # backward compatibility
                            lr_models = [lr_models]

                    for i, model in enumerate(lr_models):
                        model.eval()
                        row_bufs["lr_eval"].append(
                            {
                                "ensembling": mode,
                                "inlp_iter": i,
                                **meta,
                                **evaluate_preds(val_gt, model(val_h), mode).to_dict(),
                            }
                        )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}
