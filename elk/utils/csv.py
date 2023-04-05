from ..evaluation.evaluate_log import EvalLog
from ..logging import save_debug_log
from ..training.train_log import ElicitLog
from datasets import DatasetDict
from pathlib import Path
from typing import Iterator, Callable, TextIO, TypeVar
import csv

"""A generic log type that contains a layer field
The layer field is used to sort the logs by layer."""
Log = TypeVar("Log", EvalLog, ElicitLog)


def write_iterator_to_file(
    iterator: Iterator[Log],
    csv_columns: list[str],
    to_csv_line: Callable[[Log], list[str]],
    file: TextIO,
    debug: bool,
    dataset: DatasetDict,
    out_dir: Path,
) -> None:
    row_buf = []
    writer = csv.writer(file)
    # write a single line
    writer.writerow(csv_columns)
    try:
        for row in iterator:
            row_buf.append(row)
    finally:
        # Make sure the CSV is written even if we crash or get interrupted
        sorted_by_layer = sorted(row_buf, key=lambda x: x.layer)
        for row in sorted_by_layer:
            row = to_csv_line(row)
            writer.writerow(row)
        if debug:
            save_debug_log(dataset, out_dir)
