"""Main training loop."""

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import subgroups

from ..evaluation import Eval
from ..metrics import evaluate_preds, to_one_hot
from ..run import PreparedData, Run, LayerApplied
from ..metrics.eval import LayerOutput
from ..training.supervised import train_supervised
from ..utils.types import PromptEnsembling
from . import Classifier
from .ccs_reporter import CcsConfig, CcsReporter
from .common import FitterConfig
from .eigen_reporter import EigenFitter, EigenFitterConfig
from .multi_reporter import MultiReporter, ReporterWithInfo, SingleReporter


def evaluate_and_save(
    train_loss: float | None,
    reporter: SingleReporter | MultiReporter,
    train_dict: PreparedData,
    val_dict: PreparedData,
    lr_models: list[Classifier],
    layer: int,
):
    row_bufs = defaultdict(list)
    layer_output = []
    for ds_name in val_dict:
        val_h, val_gt, val_lm_preds = val_dict[ds_name]
        train_h, train_gt, train_lm_preds = train_dict[ds_name]
        meta = {"dataset": ds_name, "layer": layer}

        def eval_all(
            reporter: SingleReporter | MultiReporter
        ):
            val_credences = reporter(val_h)
            train_credences = reporter(train_h)
            layer_output.append(
                LayerOutput(
                    val_gt=val_gt.detach(),
                    val_credences=val_credences.detach(),
                    meta=meta,
                )
            )
            PROMPT_ENSEMBLING = "prompt_ensembling"
            for prompt_ensembling in PromptEnsembling.all():
                row_bufs["eval"].append(
                    {
                        **meta,
                        PROMPT_ENSEMBLING: prompt_ensembling.value,
                        **evaluate_preds(val_gt, val_credences, prompt_ensembling).to_dict(),
                        "train_loss": train_loss,
                    }
                )

                row_bufs["train_eval"].append(
                    {
                        **meta,
                        PROMPT_ENSEMBLING: prompt_ensembling.value,
                        **evaluate_preds(train_gt, train_credences, prompt_ensembling).to_dict(),
                        "train_loss": train_loss,
                    }
                )

                if val_lm_preds is not None:
                    row_bufs["lm_eval"].append(
                        {
                            **meta,
                            PROMPT_ENSEMBLING: prompt_ensembling.value,
                            **evaluate_preds(val_gt, val_lm_preds, prompt_ensembling).to_dict(),
                        }
                    )

                if train_lm_preds is not None:
                    row_bufs["train_lm_eval"].append(
                        {
                            **meta,
                            PROMPT_ENSEMBLING: prompt_ensembling.value,
                            **evaluate_preds(train_gt, train_lm_preds, prompt_ensembling).to_dict(),
                        }
                    )

                for lr_model_num, model in enumerate(lr_models):
                    row_bufs["lr_eval"].append(
                        {
                            **meta,
                            PROMPT_ENSEMBLING: prompt_ensembling.value,
                            "inlp_iter": lr_model_num,
                            **evaluate_preds(val_gt, model(val_h), prompt_ensembling).to_dict(),
                        }
                    )

        eval_all(reporter)

    return LayerApplied(
            layer_output, {k: pd.DataFrame(v) for k, v in row_bufs.items()}
        )


@dataclass
class Elicit(Run):
    """Full specification of a reporter training run."""

    net: FitterConfig = subgroups(
        {"ccs": CcsConfig, "eigen": EigenFitterConfig}, default="eigen"  # type: ignore
    )
    """Config for building the reporter network."""

    supervised: Literal["none", "single", "inlp", "cv"] = "single"
    """Whether to train a supervised classifier, and if so, whether to use
    cross-validation. Defaults to "single", which means to train a single classifier
    on the training data. "cv" means to use cross-validation."""

    def make_eval(self, model, eval_dataset):
        assert self.out_dir is not None
        return Eval(
            data=replace(
                self.data,
                model=model,
                datasets=(eval_dataset,),
            ),
            source=self.out_dir,
            out_dir=self.out_dir / "transfer" / eval_dataset,
            num_gpus=self.num_gpus,
            min_gpu_mem=self.min_gpu_mem,
            skip_supervised=self.supervised == "none",
            prompt_indices=self.prompt_indices,
            concatenated_layer_offset=self.concatenated_layer_offset,
            # datasets isn't needed because it's immediately overwritten
            debug=self.debug,
            disable_cache=self.disable_cache,
        )

    # Create a separate function to handle the reporter training.
    def train_and_save_reporter(
        self, device, layer, out_dir, train_dict, prompt_index=None
    ) -> ReporterWithInfo:
        print(f"train_and_save_reporter device {device}")
        (first_train_h, train_gt, _), *rest = train_dict.values()  # TODO can remove?
        (_, v, k, d) = first_train_h.shape
        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        # For a while we did support datasets with different numbers of classes, but
        # we reverted this once we switched to ConceptEraser. There are a few options
        # for re-enabling it in the future but they are somewhat complex and it's not
        # clear that it's worth it.
        if not all(other_h.shape[-2] == k for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same number of classes")

        train_loss = None
        if isinstance(self.net, CcsConfig):
            assert len(train_dict) == 1, "CCS only supports single-task training"
            reporter = CcsReporter(self.net, d, device=device, num_variants=v)
            train_loss = reporter.fit(first_train_h)
            labels = repeat(to_one_hot(train_gt, k), "n k -> n v k", v=v)
            reporter.platt_scale(labels, first_train_h)

        elif isinstance(self.net, EigenFitterConfig):
            fitter = EigenFitter(
                self.net, d, num_classes=k, num_variants=v, device=device
            )

            hidden_list, label_list = [], []
            for ds_name, (train_h, train_gt, _) in train_dict.items():
                (_, v, _, _) = train_h.shape

                # Datasets can have different numbers of variants, so we need to
                # flatten them here before concatenating
                hidden_list.append(rearrange(train_h, "n v k d -> (n v k) d"))
                label_list.append(
                    to_one_hot(repeat(train_gt, "n -> (n v)", v=v), k).flatten()
                )
                fitter.update(train_h)

            reporter = fitter.fit_streaming()
            reporter.platt_scale(
                torch.cat(label_list),
                torch.cat(hidden_list),
            )
        else:
            raise ValueError(f"Unknown reporter config type: {type(self.net)}")

        # Save reporter checkpoint to disk
        # TODO have to change this
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(reporter, out_dir / f"layer_{layer}.pt")

        return ReporterWithInfo(reporter, train_loss, prompt_index)


    def train_lr_model(self, train_dict, device, layer, out_dir) -> list[Classifier]:
        if self.supervised != "none":
            lr_models = train_supervised(
                train_dict,
                device=device,
                mode=self.supervised,
            )
            # make dir if not exists
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_models, file)
        else:
            lr_models = []

        return lr_models

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
        probe_per_prompt: bool,
    ) -> LayerApplied:
        """Train a single reporter on a single layer."""
        assert self.out_dir is not None  # TODO this is really annoying, why can it be
        # None?

        self.make_reproducible(seed=self.net.seed + layer)
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        (first_train_h, train_gt, _), *rest = train_dict.values()
        (_, v, k, d) = first_train_h.shape

        if probe_per_prompt:
            # self.prompt_indices being () actually means "all prompts"
            prompt_indices = self.prompt_indices if self.prompt_indices else range(v)
            prompt_train_dicts = [
                {
                    ds_name: (
                        train_h[:, [i], ...],
                        train_gt,
                        lm_preds[:, [i], ...] if lm_preds is not None else None,
                    )
                }
                for ds_name, (train_h, _, lm_preds) in train_dict.items()
                for i, _ in enumerate(prompt_indices)
            ]

            results = []

            for prompt_index, prompt_train_dict in zip(
                prompt_indices, prompt_train_dicts
            ):
                assert prompt_index < 100  # format i as a 2 digit string
                str_i = str(prompt_index).zfill(2)
                base = self.out_dir / "reporters" / f"prompt_{str_i}"
                reporters_path = base / "reporters"

                reporter_train_result = self.train_and_save_reporter(
                    device, layer, reporters_path, prompt_train_dict, prompt_index
                )
                results.append(reporter_train_result)

            # it is called maybe_multi_reporter because it might be a single reporter
            maybe_multi_reporter = MultiReporter(results)
            train_loss = maybe_multi_reporter.train_loss
        else:
            reporter_train_result = self.train_and_save_reporter(
                device, layer, self.out_dir / "reporters", train_dict
            )

            maybe_multi_reporter = reporter_train_result.model
            train_loss = reporter_train_result.train_loss

        lr_models = self.train_lr_model(
            train_dict, device, layer, self.out_dir / "lr_models"
        )

        return evaluate_and_save(
            train_loss, maybe_multi_reporter, train_dict, val_dict, lr_models, layer
        )
