"""Main training loop."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from einops import rearrange, repeat
from simple_parsing import subgroups
from simple_parsing.helpers.serialization import save

from ..metrics import evaluate_preds, to_one_hot
from ..run import Run
from ..training.supervised import train_supervised
from ..utils.typing import assert_type
from .ccs_reporter import CcsReporter, CcsReporterConfig
from .eigen_reporter import EigenReporter, EigenReporterConfig
from .reporter import ReporterConfig


@dataclass
class Elicit(Run):
    """Full specification of a reporter training run."""

    net: ReporterConfig = subgroups(
        {"ccs": CcsReporterConfig, "eigen": EigenReporterConfig},
        default="eigen",  # type: ignore
    )
    """Config for building the reporter network."""

    supervised: Literal["none", "single", "inlp", "cv"] = "single"
    """Whether to train a supervised classifier, and if so, whether to use
    cross-validation. Defaults to "single", which means to train a single classifier
    on the training data. "cv" means to use cross-validation."""
    num_trains: list[int] = field(default_factory=lambda: [-1])
    num_samples: int = 100

    def create_models_dir(self, out_dir: Path):
        lr_dir = None
        lr_dir = out_dir / "lr_models"
        reporter_dir = out_dir / "reporters"

        lr_dir.mkdir(parents=True, exist_ok=True)
        reporter_dir.mkdir(parents=True, exist_ok=True)

        # Save the reporter config separately in the reporter directory
        # for convenient loading of reporters later.
        save(self.net, reporter_dir / "cfg.yaml", save_dc_types=True)

        return reporter_dir, lr_dir

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
        num_platt: int = 100,
    ) -> dict[str, pd.DataFrame]:
        """Train a single reporter on a single layer."""

        self.make_reproducible(seed=self.net.seed + layer)
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        (first_train_h, _, _), *rest = train_dict.values()
        (_, _, k, d) = first_train_h.shape
        if not all(other_h.shape[-1] == d for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same hidden state size")

        # For a while we did support datasets with different numbers of classes, but
        # we reverted this once we switched to SpectralNorm. There are a few options
        # for re-enabling it in the future but they are somewhat complex and it's not
        # clear that it's worth it.
        if not all(other_h.shape[-2] == k for other_h, _, _ in rest):
            raise ValueError("All datasets must have the same number of classes")

        # carve out a subset of the training data for Platt scaling
        original_train_dict = train_dict
        train_dict = dict()
        platt_set_h = []
        platt_set_gt = []
        for ds_name, (
            o_train_h,
            o_train_gt,
            o_train_lm_preds,
        ) in original_train_dict.items():
            perm = torch.randperm(o_train_h.shape[0])
            (_, v, _, _) = o_train_h.shape
            sub_idx = perm[num_platt:]
            platt_set_idx = perm[:num_platt]
            platt_set_h.append(
                rearrange(o_train_h[platt_set_idx], "n v k d -> (n v k) d")
            )
            platt_set_gt.append(
                to_one_hot(
                    repeat(o_train_gt[platt_set_idx], "n -> (n v)", v=v), k
                ).flatten()
            )
            train_dict[ds_name] = (
                o_train_h[sub_idx],
                o_train_gt[sub_idx],
                o_train_lm_preds[sub_idx] if o_train_lm_preds is not None else None,
            )
        platt_set_h = torch.cat(platt_set_h)
        platt_set_gt = torch.cat(platt_set_gt)

        reporter_dir, lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))
        train_subs = defaultdict(list)
        train_sub_credences = defaultdict(list)
        val_credences = defaultdict(list)
        train_rand_credences = defaultdict(list)
        val_rand_credences = defaultdict(list)
        for num_train in self.num_trains:
            for i in range(self.num_samples):
                # get a random subset of the training data
                train_sub = dict()
                for ds_name, (train_h, train_gt, train_lm_preds) in train_dict.items():
                    sub_idx = torch.randperm(train_h.shape[0])[:num_platt]
                    train_h_sub = train_h[sub_idx]
                    train_gt_sub = train_gt[sub_idx]
                    train_lm_preds_sub = (
                        train_lm_preds[sub_idx] if train_lm_preds is not None else None
                    )
                    train_sub[ds_name] = (train_h_sub, train_gt_sub, train_lm_preds_sub)
                train_subs[num_train].append(train_sub)

                if isinstance(self.net, CcsReporterConfig):
                    assert (
                        len(train_dict) == 1
                    ), "CCS only supports single-task training"

                    train_h_sub, train_gt_sub, _ = train_sub.popitem()[1]
                    reporter = CcsReporter(self.net, d, device=device)
                    reporter.fit(train_h_sub, train_gt_sub)

                    (val_h, val_gt, _) = next(iter(val_dict.values()))

                    # TODO: Enable Platt scaling for CCS once normalization is fixed
                    # And add random reporter
                    # (_, v, k, _) = first_train_h.shape
                    # reporter.platt_scale(
                    #     to_one_hot(repeat(platt_gt, "n -> (n v)", v=v), k).flatten(),
                    #     rearrange(platt_h, "n v k d -> (n v k) d"),
                    # )

                elif isinstance(self.net, EigenReporterConfig):
                    reporter = EigenReporter(self.net, d, num_classes=k, device=device)

                    hidden_list, label_list = [], []
                    for ds_name, (train_h_sub, train_gt_sub, _) in train_sub.items():
                        (_, v, _, _) = train_h_sub.shape

                        # Datasets can have different numbers of variants, so we need to
                        # flatten them here before concatenating
                        hidden_list.append(
                            rearrange(train_h_sub, "n v k d -> (n v k) d")
                        )
                        label_list.append(
                            to_one_hot(
                                repeat(train_gt_sub, "n -> (n v)", v=v), k
                            ).flatten()
                        )
                        reporter.update(train_h_sub)

                    reporter.fit_streaming()
                    reporter.platt_scale(
                        platt_set_gt,
                        platt_set_h,
                    )

                    rand_reporter = EigenReporter(
                        self.net, d, num_classes=k, device=device
                    )
                    rand_reporter.weight = torch.randn_like(rand_reporter.weight)
                    rand_reporter.norm = reporter.norm
                    rand_reporter.platt_scale(
                        platt_set_gt,
                        platt_set_h,
                    )

                    train_rand_credences[num_train].append(
                        {
                            ds_name: rand_reporter(train_h)
                            for ds_name, (train_h, _, _) in train_dict.items()
                        }
                    )
                    val_rand_credences[num_train].append(
                        {
                            ds_name: rand_reporter(val_h)
                            for ds_name, (val_h, _, _) in val_dict.items()
                        }
                    )

                else:
                    raise ValueError(f"Unknown reporter config type: {type(self.net)}")

                # grab credences before throwing away the reporter
                val_credences[num_train].append(
                    {
                        ds_name: reporter(val_h)
                        for ds_name, (val_h, _, _) in val_dict.items()
                    }
                )
                train_sub_credences[num_train].append(
                    {
                        ds_name: reporter(train_h_sub)
                        for ds_name, (train_h_sub, _, _) in train_sub.items()
                    }
                )

                # Save reporter checkpoint to disk
                reporter.save(
                    reporter_dir / f"layer_{layer}_num_train_{num_train}_sample_{i}.pt"
                )

        # Fit supervised logistic regression model
        if self.supervised != "none":
            lr_models = train_supervised(
                train_dict,
                device=device,
                mode=self.supervised,
            )
            with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
                torch.save(lr_models, file)
        else:
            lr_models = []

        row_bufs = defaultdict(list)
        for ds_name in val_dict:
            val_h, val_gt, val_lm_preds = val_dict[ds_name]
            _, train_gt, train_lm_preds = train_dict[ds_name]

            for mode in ("none", "partial", "full"):
                meta = {"dataset": ds_name, "layer": layer, "ensembling": mode}

                # Log stats for each num_train
                for num_train in self.num_trains:
                    num_train_buf = defaultdict(list)

                    for i in range(self.num_samples):
                        train_h_sub, train_gt_sub, _ = train_subs[num_train][i][ds_name]
                        num_train_buf["train_eval"].append(
                            evaluate_preds(
                                train_gt_sub,
                                train_sub_credences[num_train][i][ds_name],
                                mode,
                            ).to_dict()
                        )
                        num_train_buf["eval"].append(
                            evaluate_preds(
                                val_gt,
                                val_credences[num_train][i][ds_name],
                                mode,
                            ).to_dict()
                        )
                        num_train_buf["train_rand_eval"].append(
                            evaluate_preds(
                                train_gt,
                                train_rand_credences[num_train][i][ds_name],
                                mode,
                            ).to_dict()
                        )
                        num_train_buf["rand_eval"].append(
                            evaluate_preds(
                                val_gt,
                                val_rand_credences[num_train][i][ds_name],
                                mode,
                            ).to_dict()
                        )

                    nt_dfs = {k: pd.DataFrame(v) for k, v in num_train_buf.items()}
                    # get mean, std, min, max, and 95% CI of each of
                    # auroc_estimate, acc_estimate, cal_acc_estimate, and ece

                    for key in nt_dfs:
                        stats = dict()
                        for metric in (
                            "auroc_estimate",
                            "acc_estimate",
                            "cal_acc_estimate",
                            "ece",
                        ):
                            sname = metric.replace("_estimate", "")
                            sem = float(nt_dfs[key][metric].sem())  # type: ignore
                            stats.update(
                                {
                                    f"{sname}_estimate": nt_dfs[key][metric].mean(),
                                    f"{sname}_lower": nt_dfs[key][metric].quantile(
                                        0.025
                                    ),
                                    f"{sname}_upper": nt_dfs[key][metric].quantile(
                                        0.975
                                    ),
                                    f"{sname}_std": nt_dfs[key][metric].std(),
                                    f"{sname}_ci_lower": nt_dfs[key][metric].mean()
                                    - 1.96 * sem,
                                    f"{sname}_ci_upper": nt_dfs[key][metric].mean()
                                    + 1.96 * sem,
                                    f"{sname}_sem": sem,
                                }
                            )
                        row_bufs[key].append({"num_train": num_train, **meta, **stats})

                if val_lm_preds is not None:
                    row_bufs["lm_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(val_gt, val_lm_preds, mode).to_dict(),
                        }
                    )

                if train_lm_preds is not None:
                    row_bufs["train_lm_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(train_gt, train_lm_preds, mode).to_dict(),
                        }
                    )

                for i, model in enumerate(lr_models):
                    row_bufs["lr_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            "inlp_iter": i,
                            **evaluate_preds(val_gt, model(val_h), mode).to_dict(),
                        }
                    )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}
