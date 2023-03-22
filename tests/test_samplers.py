from collections import Counter
from datasets import load_dataset, IterableDataset
from elk.extraction import BalancedBatchSampler, BalancedSampler
from elk.utils import assert_type, infer_label_column
from itertools import islice
import numpy as np


def test_output_batches_are_balanced():
    # Load an example dataset for testing
    dataset = assert_type(
        IterableDataset,
        load_dataset("super_glue", "boolq", split="train", streaming=True),
    )
    label_col = infer_label_column(dataset.features)

    # Create the BalancedBatchSampler instance
    batch_size = 32
    balanced_batch_sampler = BalancedBatchSampler(dataset, batch_size=batch_size)

    # Iterate through batches and check if they are balanced
    for batch in balanced_batch_sampler:
        counter = Counter(sample[label_col] for sample in batch)

        # Check if the output batch is balanced
        label_0_count = counter[0]
        label_1_count = counter[1]
        assert (
            label_0_count == label_1_count
        ), f"Batch is not balanced: {label_0_count}, {label_1_count}"


def test_output_is_roughly_balanced():
    # Load an example dataset for testing
    dataset = assert_type(
        IterableDataset,
        load_dataset("super_glue", "boolq", split="train", streaming=True),
    )

    col = infer_label_column(dataset.features)
    reservoir = BalancedSampler(dataset)

    # Count the number of samples for each label
    counter = Counter()
    for sample in islice(reservoir, 2000):
        counter[sample[col]] += 1

    # Check if the output is roughly balanced
    label_0_count = counter[0]
    label_1_count = counter[1]
    imbalance = abs(label_0_count - label_1_count) / (label_0_count + label_1_count)

    # Set a tolerance threshold for the imbalance ratio (e.g., 1%)
    tol = 0.01
    assert imbalance < tol, f"Imbalance ratio {imbalance} exceeded tolerance {tol}"
