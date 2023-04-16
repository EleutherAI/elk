import copy
from bisect import bisect_left, bisect_right
from functools import cache
from operator import itemgetter
from random import Random
from typing import Any, Iterable

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


def convert_span(
    offsets: list[tuple[int, int]], span: tuple[int, int]
) -> tuple[int, int]:
    """Convert `span` from string coordinates to token coordinates.

    Args:
        offsets: The offset mapping of the target tokenization.
        span: The span to convert.

    Returns:
        (start, end): The converted span.
    """
    start, end = span
    start = bisect_right(offsets, start, key=itemgetter(1))
    end = bisect_left(offsets, end, lo=start, key=itemgetter(0))
    return start, end


def get_columns_all_equal(dataset: DatasetDict) -> list[str]:
    """Get columns of a `DatasetDict`, asserting all splits have the same columns."""
    pivot, *rest = dataset.column_names.values()

    if not all(cols == pivot for cols in rest):
        raise ValueError("All splits must have the same columns")

    return pivot


def get_dataset_name(dataset: DatasetDict) -> str:
    """Get the name of a `DatasetDict`."""
    builder_name, *rest = [ds.builder_name for ds in dataset.values()]
    if not all(name == builder_name for name in rest):
        raise ValueError(
            f"All splits must have the same name; got {[builder_name, *rest]}"
        )

    config_name, *rest = [ds.config_name for ds in dataset.values()]
    if not all(name == config_name for name in rest):
        raise ValueError(
            f"All splits must have the same config name; got {[config_name, *rest]}"
        )

    include_config = config_name and has_multiple_configs(builder_name)
    return builder_name + " " + config_name if include_config else builder_name


@cache
def has_multiple_configs(ds_name: str) -> bool:
    """Return whether a dataset has multiple configs."""
    return len(get_dataset_config_names(ds_name)) > 1


def select_train_val_splits(raw_splits: Iterable[str]) -> tuple[str, str]:
    """Return splits to use for train and validation, given an Iterable of splits."""

    priorities = {
        Split.TRAIN: 0,
        Split.VALIDATION: 1,
        Split.TEST: 2,
    }

    splits = sorted(raw_splits, key=lambda k: priorities.get(k, 100))  # type: ignore
    assert len(splits) >= 2, "Must have at least two of train, val, and test splits"

    return tuple(splits[:2])


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
            f"Can't infer number of classes from label column "
            f"of type {label_feature}"
        )


def get_layers(ds: DatasetDict) -> list[int]:
    """Get a list of indices of hidden layers given a `DatasetDict`."""
    arbitrary_split = next(iter(ds.values()))
    layers = [
        int(feat[len("hidden_") :])
        for feat in arbitrary_split.features
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
