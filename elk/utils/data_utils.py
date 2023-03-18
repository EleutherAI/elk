from .typing import assert_type
from ..promptsource.templates import Template
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Split,
    Value,
    concatenate_datasets,
)
from random import Random
from typing import Optional, Iterable, Any
import numpy as np
import torch
import copy


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


def apply_template(template: Template, example: dict) -> str:
    """Concatenate question and answer if answer is not empty or whitespace."""
    q, a = template.apply(example)
    # if the jinja template already adds whitespace, don't add more
    sep = "" if not q or q[-1].isspace() or not a or a[0].isspace() else " "
    return f"{q}{sep}{a}" if a and not a.isspace() else q


def binarize(template: Template, label: int, rng: Random) -> tuple[Template, int]:
    """Binarize a template with more than 2 classes, modifying the template in-place.

    Returns:
        The new template (with only 2 answer choices).
        The new label (the index of the true answer in the new answer choices).
    """
    answer_choices = assert_type(str, template.answer_choices).split(" ||| ")
    assert len(answer_choices) > 2

    true = answer_choices[label]
    false = rng.choice([c for c in answer_choices if c != true])

    new_label = rng.choice([0, 1])
    new_template = copy.deepcopy(template)
    new_template.answer_choices = (
        f"{false} ||| {true}" if new_label else f"{true} ||| {false}"
    )

    return new_template, new_label
