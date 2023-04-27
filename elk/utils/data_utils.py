from functools import cache
from typing import Any, Iterable, Literal

from datasets import (
    ClassLabel,
    DatasetDict,
    Features,
    Value,
    get_dataset_config_names,
)

from .typing import assert_type


def get_columns_all_equal(dataset: DatasetDict) -> list[str]:
    """Get columns of a `DatasetDict`, asserting all splits have the same columns."""
    pivot, *rest = dataset.column_names.values()

    if not all(cols == pivot for cols in rest):
        raise ValueError("All splits must have the same columns")

    return pivot


def get_split_priority(split: str) -> int:
    """Return an integer indicating how "test-like" a split is given its name."""
    if split.startswith("train"):
        return 0
    elif split.startswith("val"):
        return 1
    elif split.startswith("test"):
        return 2

    return 3


@cache
def has_multiple_configs(ds_name: str) -> bool:
    """Return whether a dataset has multiple configs."""
    return len(get_dataset_config_names(ds_name)) > 1


def select_split(raw_splits: Iterable[str], split_type: Literal["train", "val"]) -> str:
    """Return the train or validation split to use, given an Iterable of splits."""
    assert split_type in ("train", "val"), f"Invalid split type: {split_type}"

    # Note we use the alphabetical order of the splits as a tiebreaker.
    sorted_splits = sorted(raw_splits, key=lambda k: (get_split_priority(k), k))
    if not sorted_splits:
        raise ValueError("No splits found!")
    elif len(sorted_splits) == 1:
        return sorted_splits[0]
    else:
        return sorted_splits[0] if split_type == "train" else sorted_splits[1]


def select_train_val_splits(raw_splits: Iterable[str]) -> tuple[str, str]:
    """Return splits to use for train and validation, given an Iterable of splits."""

    splits = sorted(raw_splits, key=lambda k: (get_split_priority(k), k))
    assert len(splits) >= 2, "Must have at least two of train, val, and test splits"

    return tuple(splits[:2])


def infer_label_column(features: Features) -> str:
    """Return the unique `ClassLabel` column, or "label" if it's of a suitable dtype.

    Returns:
        The name of the unique label column.
    Raises:
        ValueError: If it's unclear what the label column is.
    """
    label_cols = [
        col for col, dtype in features.items() if isinstance(dtype, ClassLabel)
    ]
    if not label_cols:
        # One more heuristic: if there's a column just named "label" with a reasonable
        # dtype, use that.
        col = features.get("label")
        if not col:
            raise ValueError(
                "None of the columns in the dataset are obviously the label column; "
                "please specify label_column in the prompt template yaml file."
            )

        import pyarrow as pa

        if pa.types.is_integer(col.pa_type) or col.dtype in ("bool", "string"):
            return "label"
        else:
            # We don't support floats, timestamps, bytes, containers, etc.
            raise ValueError(
                f"Column 'label' has unsupported dtype {col.dtype}; please specify "
                "a different label_column in the prompt template yaml file."
            )

    elif len(label_cols) > 1:
        raise ValueError(
            f"Dataset has multiple label columns {label_cols}; specify label_column "
            "in the prompt template yaml to disambiguate"
        )
    else:
        return assert_type(str, label_cols[0])


def infer_num_classes(label_feature: Any) -> int:
    """Return the number of classes in a `Dataset`.

    Returns:
        The number of classes.
    Raises:
        ValueError: If the label column is not a `ClassLabel` or `Value('bool')`.
    """
    if isinstance(label_feature, ClassLabel):
        # We piggyback on the ClassLabel feature type to get the number of classes
        return label_feature.num_classes  # type: ignore
    elif isinstance(label_feature, Value) and label_feature.dtype == "bool":
        return 2
    else:
        return -1


def get_layer_indices(ds: DatasetDict) -> list[int]:
    """Return the indices of the layers from which the hiddens have been extracted."""
    # Dataset has a bunch of columns of the form "hidden_0", "hidden_1", etc.
    # str.removeprefix() is a no-op if the prefix isn't present
    suffixes = (col.removeprefix("hidden_") for col in get_columns_all_equal(ds))

    # Convert to the suffixes that are integral to ints, then sort them
    return sorted(int(suffix) for suffix in suffixes if suffix.isdigit())
