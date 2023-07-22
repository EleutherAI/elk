from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from simple_parsing.helpers import field
from torch import Tensor

from ..files import elk_reporter_dir
from ..metrics import evaluate_preds
from ..run import Run
from ..training.multi_reporter import MultiReporter, SingleReporter
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
        self, layer: int, devices: list[str], world_size: int, probe_per_prompt: bool
    ) -> tuple[dict, list[list[dict[str, Tensor | Any]]]]:
        """Evaluate a single reporter on a single layer."""
        device = self.get_device(devices, world_size)
        val_output = self.prepare_data(device, layer, "val")

        experiment_dir = elk_reporter_dir() / self.source

        def load_reporter() -> SingleReporter | MultiReporter:
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
            reporter: SingleReporter | MultiReporter,
            prompt_index: int | Literal["multi"] | None = None,
            i: int = 0,
        ):
            prompt_index_dict = (
                {"prompt_index": prompt_index} if prompt_index is not None else {}
            )
            layer_outputs = []
            for ds_name, (val_h, val_gt, val_lm_preds) in val_output.items():
                meta = {"dataset": ds_name, "layer": layer}
                if isinstance(prompt_index, int):
                    val_credences = reporter(val_h[:, [i], :, :])
                else:
                    val_credences = reporter(val_h)

                layer_outputs.append(
                    {**meta, "val_gt": val_gt, "val_credences": val_credences}
                )

                for prompt_ensembling in PromptEnsembling.all():
                    row_bufs["eval"].append(
                        {
                            **meta,
                            "prompt_ensembling": prompt_ensembling.value,
                            **evaluate_preds(
                                val_gt, val_credences, prompt_ensembling
                            ).to_dict(),
                            **prompt_index_dict,
                        }
                    )

                    if val_lm_preds is not None:
                        row_bufs["lm_eval"].append(
                            {
                                **meta,
                                "prompt_ensembling": prompt_ensembling.value,
                                **evaluate_preds(
                                    val_gt, val_lm_preds, prompt_ensembling
                                ).to_dict(),
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
                                        "prompt_ensembling": prompt_ensembling.value,
                                        "inlp_iter": i,
                                        **meta,
                                        **evaluate_preds(
                                            val_gt, model(val_h), prompt_ensembling
                                        ).to_dict(),
                                    }
                                )
                return layer_outputs

        layer_output = []
        for ds_name, (val_h, val_gt, _) in val_output.items():
            meta = {"dataset": ds_name, "layer": layer}

            val_credences = reporter(val_h)
            layer_output.append(
                {**meta, "val_gt": val_gt, "val_credences": val_credences}
            )
            for prompt_ensembling in PromptEnsembling.all():
                row_bufs["eval"].append(
                    {
                        **meta,
                        "prompt_ensembling": prompt_ensembling.value,
                        **evaluate_preds(
                            val_gt, val_credences, prompt_ensembling
                        ).to_dict(),
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
                                "prompt_ensembling": prompt_ensembling.value,
                                "inlp_iter": i,
                                **meta,
                                **evaluate_preds(
                                    val_gt, model(val_h), prompt_ensembling
                                ).to_dict(),
                            }
                        )

        layer_outputs = []
        if isinstance(reporter, MultiReporter):
            # eg.
            # prompt_indices       = 0 1 5 9
            # i of the data passed = 0 1 2 3
            for i, res in enumerate(reporter.reporter_w_infos):
                layer_outputs.append(eval_all(res.model, res.prompt_index, i))
            layer_outputs.append(eval_all(reporter, "multi"))
        else:
            layer_outputs.append(eval_all(reporter))

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}, layer_outputs
