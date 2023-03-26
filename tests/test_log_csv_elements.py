from elk.evaluation.evaluate_log import EvalLog
from elk.training.reporter import EvalResult
from elk.training.train_log import ElicitLog


def test_eval_log_csv_number_elements():
    log = EvalLog(
        layer=1,
        eval_result=EvalResult(
            acc=1.0,
            cal_acc=1.0,
            auroc=1.0,
            ece=1.0,
        ),
    )
    csv_columns = EvalLog.csv_columns()
    csv_values = log.to_csv_line()
    assert len(csv_columns) == len(
        csv_values
    ), "Number of columns and values should be the same"


def test_elicit_log_csv_number_elements():
    log = ElicitLog(
        layer=1,
        train_loss=1.0,
        eval_result=EvalResult(
            acc=0.0,
            ece=0.0,
            cal_acc=0.0,
            auroc=0.0,
        ),
        pseudo_auroc=0.0,
    )
    csv_columns = ElicitLog.csv_columns(skip_baseline=True)
    csv_values = log.to_csv_line(skip_baseline=True)
    assert len(csv_columns) == len(
        csv_values
    ), "Number of columns and values should be the same"
    csv_columns_not_skipped = ElicitLog.csv_columns(skip_baseline=False)
    csv_values_not_skipped = log.to_csv_line(skip_baseline=False)
    assert len(csv_columns_not_skipped) == len(
        csv_values_not_skipped
    ), "Number of columns and values should be the same"
