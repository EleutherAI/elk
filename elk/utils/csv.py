import csv
from pathlib import Path
from typing import Iterator, Callable, TextIO, TypeVar

from datasets import DatasetDict

from elk.evaluation.evaluate_log import EvalLog
from elk.logging import save_debug_log
from elk.training.train_log import ElicitLog
import os
import pickle

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
        for i, row in enumerate(sorted_by_layer):
            # as above except format the layer number as a 4 digits
            if isinstance(row, EvalLog):
                pickle.dump(row.proposition_results, open(os.path.join(out_dir, f"layer_{i:04d}_propositions.pkl"), "wb"))
            row = to_csv_line(row)
            writer.writerow(row)
        if debug:
            save_debug_log(dataset, out_dir)
