from dataclasses import dataclass

from elk.training.reporter import EvalResult


@dataclass
class EvalLog:
    """The result of running eval on a layer of a dataset"""

    layer: int
    eval_result: EvalResult
    proposition_results: dict

    @staticmethod
    def csv_columns() -> list[str]:
        return ["layer", "acc", "cal_acc", "auroc", "ece"]

    def to_csv_line(self) -> list[str]:
        items = [
            self.layer,
            self.eval_result.acc,
            self.eval_result.cal_acc,
            self.eval_result.auroc,
            self.eval_result.ece,
        ]
        return [
            f"{item:.4f}" if isinstance(item, float) else str(item) for item in items
        ]
