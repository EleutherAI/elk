from .reporter import EvalResult
from dataclasses import dataclass
from typing import Optional


@dataclass
class ElicitLog:
    """The result of running elicit on a layer of a dataset"""

    layer: int
    pseudo_auroc: float
    train_loss: float
    eval_result: EvalResult

    # Only available when the LM is autoregressive
    lm_auroc: Optional[float] = None
    lm_acc: Optional[float] = None

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
            "lm_auroc",
            "lm_acc",
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
        if self.lm_auroc is not None and self.lm_acc is not None:
            items += [self.lm_auroc, self.lm_acc]
        if not skip_baseline:
            items += [self.lr_auroc, self.lr_acc]

        return [
            f"{item:.4f}" if isinstance(item, float) else str(item) for item in items
        ]
