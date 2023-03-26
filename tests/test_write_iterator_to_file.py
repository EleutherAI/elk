import csv
from pathlib import Path
from typing import Iterator

from datasets import DatasetDict

from elk.utils.csv import write_iterator_to_file
from elk.training.reporter import EvalResult
from elk.training.train_log import ElicitLog


def test_write_iterator_to_file(tmp_path: Path):
    items: list[ElicitLog] = [
        ElicitLog(
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
    ]
    iterator = iter(items)
    csv_columns = ElicitLog.csv_columns(skip_baseline=True)
    to_csv_line = lambda x: x.to_csv_line(skip_baseline=True)
    # Write the CSV file
    with open(tmp_path / "test.csv", "w") as f:
        write_iterator_to_file(
            iterator=iterator,
            file=f,
            debug=False,
            dataset=DatasetDict(),
            out_dir=tmp_path,
            csv_columns=csv_columns,
            to_csv_line=to_csv_line,
        )
    # Read the CSV file
    with open(tmp_path / "test.csv", "r") as f:
        reader = csv.reader(f)
        # assert that the first line is the header
        assert next(reader) == csv_columns
        # assert that the second line is the data
        assert next(reader) == to_csv_line(items[0])


def test_write_iterator_to_file_crash(tmp_path: Path):
    first_layer_log = ElicitLog(
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

    second_layer_log = ElicitLog(
        layer=2,
        train_loss=1.0,
        eval_result=EvalResult(
            acc=0.0,
            ece=0.0,
            cal_acc=0.0,
            auroc=0.0,
        ),
        pseudo_auroc=0.0,
    )

    def iterator() -> Iterator[ElicitLog]:
        for i in range(3):
            if i == 0:
                yield first_layer_log
            elif i == 1:
                yield second_layer_log
            # on the third iteration, raise an ValueError
            # We should still be able to write the first two layers
            if i == 2:
                raise ValueError()

    csv_columns = ElicitLog.csv_columns(skip_baseline=True)
    to_csv_line = lambda x: x.to_csv_line(skip_baseline=True)
    # Write the CSV file
    try:
        with open(tmp_path / "test.csv", "w") as f:
            write_iterator_to_file(
                iterator=iterator(),
                file=f,
                debug=False,
                dataset=DatasetDict(),
                out_dir=tmp_path,
                csv_columns=csv_columns,
                to_csv_line=to_csv_line,
            )
    except ValueError:
        pass

    # Read the CSV file
    with open(tmp_path / "test.csv", "r") as f:
        reader = csv.reader(f)
        # assert that the first line is the header
        assert next(reader) == csv_columns
        # assert that the second line has the first layer
        assert next(reader) == to_csv_line(first_layer_log)
        # assert that the third line has the second layer
        assert next(reader) == to_csv_line(second_layer_log)
