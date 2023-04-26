import copy
from functools import cache
from random import Random
from typing import Any, Iterable, Literal

from datasets import (
    ClassLabel,
    DatasetDict,
    Features,
    Split,
    Value,
    get_dataset_config_names,
)

from ..promptsource.templates import Template
from .typing import assert_type

# Lower values are more "train-like" and higher values are more "test-like".
PRIORITIES = {
    Split.TRAIN: 0,
    Split.VALIDATION: 1,
    Split.TEST: 2,
}


def get_columns_all_equal(dataset: DatasetDict) -> list[str]:
    """Get columns of a `DatasetDict`, asserting all splits have the same columns."""
    pivot, *rest = dataset.column_names.values()

    if not all(cols == pivot for cols in rest):
        raise ValueError("All splits must have the same columns")

    return pivot


@cache
def has_multiple_configs(ds_name: str) -> bool:
    """Return whether a dataset has multiple configs."""
    return len(get_dataset_config_names(ds_name)) > 1


def select_split(raw_splits: Iterable[str], split_type: Literal["train", "val"]) -> str:
    """Return the train or validation split to use, given an Iterable of splits."""
    assert split_type in ("train", "val")

    reduce_fn = min if split_type == "train" else max
    return reduce_fn(raw_splits, key=lambda k: PRIORITIES.get(k, 100))  # type: ignore


def select_train_val_splits(raw_splits: Iterable[str]) -> tuple[str, str]:
    """Return splits to use for train and validation, given an Iterable of splits."""

    splits = sorted(raw_splits, key=lambda k: PRIORITIES.get(k, 100))  # type: ignore
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
        raise ValueError(
            f"Can't infer number of classes from label column of type {label_feature}. "
            f"Please update the num_classes field in the prompt template yaml file."
        )


def get_layers(ds: DatasetDict) -> list[int]:
    """Get a list of indices of hidden layers given a `DatasetDict`."""
    train, _ = select_train_val_splits(ds.keys())
    layers = [
        int(feat[len("hidden_") :])
        for feat in ds[train].features
        if feat.startswith("hidden_")
    ]
    return layers


def binarize(template: Template, label: int, new_label: int, rng: Random) -> Template:
    """Binarize a template with >2 answer choices, returning a new template and label.
    Returns:
        `new_template`:
            A deepcopy of the original template with with 2 answer choices, one of
            which is the true answer and the other is a random false answer.
        `new_label`:
            the index of the true answer into `new_template.answer_choices`
    """

    # TODO: it would be nice in the future to binarize exhaustively so we're not
    # cheating here (since this step requires a label). e.g. this function would
    # also take a candidate answer and the template would ask whether the candidate
    # answer is true or false. This would require rewriting the jinja templates though.
    answer_choices = assert_type(str, template.answer_choices).split(" ||| ")
    assert len(answer_choices) > 2

    true = answer_choices[label]
    false = rng.choice([c for c in answer_choices if c != true])

    assert new_label in (0, 1)

    new_template = copy.deepcopy(template)
    new_template.answer_choices = (
        f"{false} ||| {true}" if new_label else f"{true} ||| {false}"
    )

    return new_template
