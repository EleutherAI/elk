"""Main training loop."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from ..extraction import Extract
from ..metrics import evaluate_preds, get_logprobs
from ..run import Run
from ..training.supervised import train_supervised
from ..utils.typing import assert_type


@dataclass
class Elicit(Run):
    """Full specification of a reporter training run."""

    seed: int = 42

    supervised: Literal["single", "inlp", "cv"] = "single"
    """Whether to train a supervised classifier, and if so, whether to use
    cross-validation. Defaults to "single", which means to train a single classifier
    on the training data. "cv" means to use cross-validation."""

    erase_paraphrases: bool = False
    """Whether to use LEACE to erase the paraphrase dimensions before training the
    classifier."""

    max_inlp_iter: int | None = None
    """Maximum number of iterations for Iterative Nullspace Projection (INLP)."""

    @staticmethod
    def default():
        return Elicit(
            data=Extract(
                model="<placeholder>",
                datasets=("<placeholder>",),
            )
        )

    def create_models_dir(self, out_dir: Path):
        lr_dir = out_dir / "lr_models"

        lr_dir.mkdir(parents=True, exist_ok=True)

        return lr_dir

    def apply_to_layer(
        self,
        layer: int,
        devices: list[str],
        world_size: int,
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        """Train a single reporter on a single layer."""

        self.make_reproducible(seed=self.seed + layer)
        device = self.get_device(devices, world_size)

        train_dict = self.prepare_data(device, layer, "train")
        val_dict = self.prepare_data(device, layer, "val")

        first_train_data, *rest = train_dict.values()
        (_, v, d) = first_train_data.hiddens.shape
        if not all(other_data.hiddens.shape[-1] == d for other_data in rest):
            raise ValueError("All datasets must have the same hidden state size")

        lr_dir = self.create_models_dir(assert_type(Path, self.out_dir))

        # Fit supervised logistic regression model

        lr_models = train_supervised(
            train_dict,
            erase_paraphrases=self.erase_paraphrases,
            device=device,
            mode=self.supervised,
            max_inlp_iter=self.max_inlp_iter,
        )
        with open(lr_dir / f"layer_{layer}.pt", "wb") as file:
            torch.save(lr_models, file)

        out_logprobs = defaultdict(dict)
        row_bufs = defaultdict(list)
        for ds_name in val_dict:
            val, train = val_dict[ds_name], train_dict[ds_name]
            meta = {"dataset": ds_name, "layer": layer}

            if self.save_logprobs:
                out_logprobs[ds_name] = dict(
                    row_ids=val.row_ids,
                    variant_ids=val.variant_ids,
                    texts=val.texts,
                    labels=val.labels,
                    lm=dict(),
                    lr=dict(),
                )

            for mode in ("none", "full"):
                if val.lm_log_odds is not None:
                    if self.save_logprobs:
                        out_logprobs[ds_name]["lm"][mode] = (
                            get_logprobs(val.lm_log_odds, mode).detach().cpu()
                        )

                    row_bufs["lm_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            **evaluate_preds(
                                val.labels, val.lm_log_odds, mode
                            ).to_dict(),
                        }
                    )

                if self.save_logprobs:
                    out_logprobs[ds_name]["lr"][mode] = dict()

                for i, model in enumerate(lr_models):
                    model.eval()
                    val_log_odds = model(val.hiddens)
                    train_log_odds = model(train.hiddens)

                    if self.save_logprobs:
                        out_logprobs[ds_name]["lr"][mode][i] = (
                            get_logprobs(val_log_odds, mode).detach().cpu()
                        )

                    row_bufs["lr_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            "inlp_iter": i,
                            **evaluate_preds(val.labels, val_log_odds, mode).to_dict(),
                        }
                    )

                    row_bufs["train_lr_eval"].append(
                        {
                            **meta,
                            "ensembling": mode,
                            "inlp_iter": i,
                            **evaluate_preds(
                                train.labels, train_log_odds, mode
                            ).to_dict(),
                        }
                    )

        return {k: pd.DataFrame(v) for k, v in row_bufs.items()}, out_logprobs
