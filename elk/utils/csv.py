import csv
from pathlib import Path
from typing import Iterator, Callable, TextIO

from datasets import DatasetDict

from elk.logging import save_debug_log
from elk.run import Log


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
