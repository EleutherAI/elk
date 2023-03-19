from .typing import assert_type
from bisect import bisect_left, bisect_right
from datasets import ClassLabel, Dataset, DatasetDict, Features, concatenate_datasets
from operator import itemgetter
from random import Random
from typing import Optional
import numpy as np
import torch


def compute_class_balance(
    dataset: Dataset,
    num_classes: int,
    label_column: Optional[str] = None,
) -> np.ndarray:
    """Compute the class balance of a `Dataset`."""

    features = dataset.features
    name = dataset.info.builder_name
    if label_column is None:
        label_column = infer_label_column(dataset.features)
    elif label_column not in features:
        raise ValueError(f"{name} has no column '{label_column}'")

    class_sizes = np.bincount(dataset[label_column], minlength=num_classes)

    if not np.all(class_sizes > 0):
        missing = np.flatnonzero(class_sizes == 0).tolist()
        raise ValueError(f"{name} has missing classes: {missing}")

    return class_sizes


def get_columns_all_equal(dataset: DatasetDict) -> list[str]:
    """Get columns of a `DatasetDict`, asserting all splits have the same columns."""
    pivot, *rest = dataset.column_names.values()

    if not all(cols == pivot for cols in rest):
        raise ValueError("All splits must have the same columns")

    return pivot


def held_out_split(dataset: DatasetDict) -> Dataset:
    """Return the validation set if it exits, otherwise the test set."""
    if "validation" in dataset:
        return dataset["validation"]
    elif "test" in dataset:
        return dataset["test"]
    else:
        raise ValueError("No validation or test split found")


def infer_label_column(features: Features) -> str:
    """Return the unique `ClassLabel` column in a `Dataset`.

    Returns:
        The name of the unique label column.
    Raises:
        ValueError: If there are no `ClassLabel` columns, or if there are multiple.
    """
    label_cols = [
        col for col, dtype in features.items() if isinstance(dtype, ClassLabel)
    ]
    if not label_cols:
        raise ValueError("Dataset has no label column")
    elif len(label_cols) > 1:
        raise ValueError(
            f"Dataset has multiple label columns {label_cols}; specify "
            f"label_column to disambiguate"
        )
    else:
        return assert_type(str, label_cols[0])


def undersample(
    dataset: Dataset, rng: Random, num_classes: int, label_column: Optional[str] = None
) -> Dataset:
    """Undersample a `Dataset` to the smallest class size."""
    label_column = label_column or infer_label_column(dataset.features)
    class_sizes = compute_class_balance(dataset, num_classes, label_column)
    smallest_size = class_sizes.min()

    # First group the active split by class
    strata = (
        dataset.filter(lambda ex: ex[label_column] == i)
        for i in range(len(class_sizes))
    )
    # Then randomly sample `smallest_size` examples from each class and merge
    strata = [
        stratum.select(rng.sample(range(len(stratum)), k=smallest_size))
        for stratum in strata
    ]
    dataset = assert_type(Dataset, concatenate_datasets(strata))

    # Sanity check that we successfully balanced the classes
    class_sizes = np.bincount(dataset[label_column], minlength=len(class_sizes))
    assert np.all(class_sizes == smallest_size)

    return dataset


def float32_to_int16(x: torch.Tensor) -> torch.Tensor:
    """Converts float32 to float16, then reinterprets as int16."""
    return x.type(torch.float16).view(torch.int16)


def int16_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Converts int16 to float16, then reinterprets as float32."""
    return x.view(torch.float16).type(torch.float32)
