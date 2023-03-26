from dataclasses import dataclass
from typing import Optional, TypeVar

from simple_parsing import Serializable

from elk.training.reporter import EvalResult


@dataclass
class ElicitLog:
    """The result of running elicit on a layer of a dataset"""

    layer: int
    train_loss: float
    eval_result: EvalResult
    pseudo_auroc: float
    # Only available if reporting baseline
    lr_auroc: Optional[float] = None
    # Only available if reporting baseline
    lr_acc: Optional[float] = None

    @staticmethod
    def csv_columns(skip_baseline: bool) -> list[str]:
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
class EvalLog:
    """The result of running eval on a layer of a dataset"""
    layer: int
    eval_result: EvalResult

    @staticmethod
    def csv_columns() -> list[str]:
        return ["layer", "acc", "cal_acc", "auroc"]

    def to_csv_line(self) -> list[str]:
        items = [
            self.layer,
            self.eval_result.acc,
            self.eval_result.cal_acc,
            self.eval_result.auroc,
        ]
        return [f"{item:.4f}" for item in items if isinstance(item, float)]


StatResult = TypeVar("StatResult", ElicitLog, EvalLog)
