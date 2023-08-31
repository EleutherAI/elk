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
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.source

        reporter_path = experiment_dir / "reporters" / f"layer_{layer}.pt"
        reporter = torch.load(reporter_path, map_location=device)

        out_logprobs = defaultdict(dict)
        row_bufs = defaultdict(list)
        for ds_name, val_data in val_output.items():
            meta = {"dataset": ds_name, "layer": layer}

            if self.save_logprobs:
                out_logprobs[ds_name]["texts"] = val_data.text_questions
                out_logprobs[ds_name]["labels"] = val_data.labels.cpu()
                out_logprobs[ds_name]["reporter"] = dict()
                out_logprobs[ds_name]["lr"] = dict()
                out_logprobs[ds_name]["lm"] = dict()

            val_credences = reporter(val_data.hiddens)
            for mode in ("none", "partial", "full"):
                if self.save_logprobs:
                    out_logprobs[ds_name]["reporter"][mode] = get_logprobs(
                        val_credences, mode
                    ).cpu()
                    out_logprobs[ds_name]["lm"][mode] = (
                        get_logprobs(val_data.lm_preds, mode).cpu()
                        if val_data.lm_preds is not None
                        else None
                    )

                row_bufs["eval"].append(
                    {
                        **meta,
                        "ensembling": mode,
                        **evaluate_preds(
                            val_data.labels, val_credences, mode
                        ).to_dict(),
                    }
                )

                if val_data.lm_preds is not None:
                    row_bufs["lm_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(
                                val_data.labels, val_data.lm_preds, mode
                            ).to_dict(),
                        }
                    )

                lr_dir = experiment_dir / "lr_models"
                if not self.skip_supervised and lr_dir.exists():
                    if self.save_logprobs:
                        out_logprobs[ds_name]["lr"][mode] = dict()

                    with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
                        lr_models = torch.load(f, map_location=device)
                        if not isinstance(lr_models, list):  # backward compatibility
                            lr_models = [lr_models]

                    for i, model in enumerate(lr_models):
                        model.eval()
                        val_lr_credences = model(val_data.hiddens)
                        if self.save_logprobs:
                            out_logprobs[ds_name]["lr"][mode][i] = get_logprobs(
                                val_lr_credences, mode
                            ).cpu()
                        row_bufs["lr_eval"].append(
                            {
                                "ensembling": mode,
                                "inlp_iter": i,
                                **meta,
                                **evaluate_preds(
                                    val_data.labels, val_lr_credences, mode
                                ).to_dict(),
                            }
                        )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}, out_logprobs
