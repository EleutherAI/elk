from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from simple_parsing.helpers import field

from ..files import elk_reporter_dir
from ..metrics import evaluate_preds, get_logprobs
from ..run import Run
from ..utils import Color


@dataclass(kw_only=True)
class Eval(Run):
    """Full specification of a reporter evaluation run."""

    source: Path = field(positional=True)

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
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.source

        lr_dir = experiment_dir / "lr_models"
        with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
            lr_models = torch.load(f, map_location=device)
            if not isinstance(lr_models, list):  # backward compatibility
                lr_models = [lr_models]

        out_logprobs = defaultdict(dict)
        row_bufs = defaultdict(list)
        for ds_name, val_data in val_output.items():
            meta = {"dataset": ds_name, "layer": layer}

            if self.save_logprobs:
                out_logprobs[ds_name] = dict(
                    row_ids=val_data.row_ids,
                    variant_ids=val_data.variant_ids,
                    texts=val_data.texts,
                    labels=val_data.labels,
                    lm=dict(),
                    lr=dict(),
                )
            for mode in ("none", "full"):
                if val_data.lm_log_odds is not None:
                    if self.save_logprobs:
                        out_logprobs[ds_name]["lm"][mode] = get_logprobs(
                            val_data.lm_log_odds, mode
                        ).cpu()
                    row_bufs["lm_eval"].append(
                        {
                            "ensembling": mode,
                            **meta,
                            **evaluate_preds(
                                val_data.labels, val_data.lm_log_odds, mode
                            ).to_dict(),
                        }
                    )

                if self.save_logprobs:
                    out_logprobs[ds_name]["lr"][mode] = dict()

                for i, model in enumerate(lr_models):
                    model.eval()
                    val_log_odds = model(val_data.hiddens)
                    if self.save_logprobs:
                        out_logprobs[ds_name]["lr"][mode][i] = get_logprobs(
                            val_log_odds, mode
                        ).cpu()
                    row_bufs["lr_eval"].append(
                        {
                            "ensembling": mode,
                            "inlp_iter": i,
                            **meta,
                            **evaluate_preds(
                                val_data.labels, val_log_odds, mode
                            ).to_dict(),
                        }
                    )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}, out_logprobs
