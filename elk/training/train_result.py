from dataclasses import dataclass
from typing import Optional

from simple_parsing import Serializable

from elk.training.reporter import EvalResult


@dataclass
class ElicitStatResult:
    """The result of evaluating a reporter on a dataset."""

    layer: int
    train_loss: float
    eval_result: EvalResult
    pseudo_auroc: float
    # Only available if reporting baseline
    lr_auroc: Optional[float] = None
    # Only available if reporting baseline
    lr_acc: Optional[float] = None

    cols = ["layer", "loss", "acc", "cal_acc", "auroc"]
    @staticmethod
    def to_csv_columns(skip_baseline: bool) -> list[str]:
        """Return a CSV header with the column names."""
        cols = [
            "layer",
            "pseudo_auroc",
            "train_loss",
            "acc",
            "cal_acc",
            "auroc",
            "ece",
        ]
        if not skip_baseline:
            cols += ["lr_auroc", "lr_acc"]
        return cols



    def to_csv_line(self, skip_baseline: bool) -> list[str]:
        """Return a CSV line with the evaluation results."""
        items = [
            self.layer,
            self.pseudo_auroc,
            self.train_loss,
            self.eval_result.acc,
            self.eval_result.cal_acc,
            self.eval_result.auroc,
            self.eval_result.ece,
        ]
        if not skip_baseline:
            items += [self.lr_auroc, self.lr_acc]
        # TODO: 4f for floats?
        return [f"{item:.4f}" for item in items if isinstance(item, float)]


@dataclass
class EvalStatResult:
    ...
