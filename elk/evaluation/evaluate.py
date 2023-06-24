from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from simple_parsing.helpers import field

from ..files import elk_reporter_dir
from ..metrics import evaluate_preds
from ..run import Run
from ..training import Reporter
from ..utils import Color
from ..utils.types import PromptEnsembling


@dataclass(kw_only=True)
class Eval(Run):
    """Full specification of a reporter evaluation run."""

    source: Path = field(positional=True)
    skip_supervised: bool = False

    def __post_init__(self):
        # Set our output directory before super().execute() does
        if not self.out_dir:
            root = elk_reporter_dir() / self.source
            self.out_dir = root / "transfer" / "+".join(self.data.datasets)

    def execute(self, highlight_color: Color = "cyan"):
        return super().execute(highlight_color, split_type="val")

    @torch.inference_mode()
    def apply_to_layer(
        self, layer: int, devices: list[str], world_size: int
    ) -> tuple[dict[str, pd.DataFrame], list[dict]]:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter = Reporter.load(reporter_path, map_location=device)
        reporter.eval()

        row_bufs = defaultdict(list)

        layer_outputs = []
        for ds_name, (val_h, val_gt, _) in val_output.items():
            meta = {"dataset": ds_name, "layer": layer}

            val_credences = reporter(val_h)
            layer_outputs.append(
                {**meta, "val_gt": val_gt, "val_credences": val_credences}
            )
            for ensembling in PromptEnsembling.all():
                row_bufs["eval"].append(
                    {
                        **meta,
                        "ensembling": ensembling.value,
                        **evaluate_preds(val_gt, val_credences, ensembling).to_dict(),
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
                                "ensembling": ensembling.value,
                                "inlp_iter": i,
                                **meta,
                                **evaluate_preds(
                                    val_gt, model(val_h), ensembling
                                ).to_dict(),
                            }
                        )

        return ({k: pd.DataFrame(v) for k, v in row_bufs.items()}, layer_outputs)
