from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from simple_parsing.helpers import field

from ..files import elk_reporter_dir
from ..metrics import evaluate_preds
from ..run import Run
from ..training.multi_reporter import AnyReporter, MultiReporter
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
        self, layer: int, devices: list[str], world_size: int, probe_per_prompt: bool
    ) -> dict[str, pd.DataFrame]:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        val_output = {
            ds_name: (
                train_h[:, self.prompt_indices, ...],
                train_gt,
                lm_preds[:, self.prompt_indices, ...] if lm_preds is not None else None,
            )
            for ds_name, (train_h, train_gt, lm_preds) in val_output.items()
        }

        experiment_dir = elk_reporter_dir() / self.source

        def load_reporter() -> AnyReporter | MultiReporter:
            # check if experiment_dir / "reporters" has .pt files
            first = next((experiment_dir / "reporters").iterdir())
            if not first.suffix == ".pt":
                return MultiReporter.load(
                    experiment_dir / "reporters", layer, device=device
                )
            else:
                path = experiment_dir / "reporters" / f"layer_{layer}.pt"
                return torch.load(path, map_location=device)

        reporter = load_reporter()

        row_bufs = defaultdict(list)

        def eval_all(
            reporter: AnyReporter | MultiReporter,
            prompt_index: int | Literal["multi"] | None = None,
        ):
            prompt_index = (
                {"prompt_index": prompt_index} if prompt_index is not None else {}
            )
            for ds_name, (val_h, val_gt, _) in val_output.items():
                meta = {"dataset": ds_name, "layer": layer}

                val_credences = reporter(val_h)
                for mode in ("none", "partial", "full"):
                    row_bufs["eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(val_gt, val_credences, mode).to_dict(),
                            **prompt_index,
                        }
                    )

                    lr_dir = experiment_dir / "lr_models"
                    if not self.skip_supervised and lr_dir.exists():
                        with open(lr_dir / f"layer_{layer}.pt", "rb") as f:
                            lr_models = torch.load(f, map_location=device)
                            if not isinstance(
                                lr_models, list
                            ):  # backward compatibility
                                lr_models = [lr_models]

                        for i, model in enumerate(lr_models):
                            model.eval()
                            row_bufs["lr_eval"].append(
                                {
                                    "ensembling": mode,
                                    "inlp_iter": i,
                                    **meta,
                                    **evaluate_preds(
                                        val_gt, model(val_h), mode
                                    ).to_dict(),
                                }
                            )

        if isinstance(reporter, MultiReporter):
            for prompt_index, single_reporter in enumerate(reporter.reporters):
                eval_all(single_reporter, prompt_index)
            eval_all(reporter, "multi")
        else:
            eval_all(reporter)

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}
