from collections import Counter
from itertools import islice
from random import Random

from datasets import IterableDataset, load_dataset

from elk.extraction import BalancedSampler, FewShotSampler
from elk.utils import assert_type, infer_label_column


def test_output_batches_are_balanced():
    # Load an example dataset for testing
    dataset = assert_type(
        IterableDataset,
        load_dataset("super_glue", "boolq", split="train", streaming=True),
    )
    label_col = infer_label_column(dataset.features)

    # Start with an even number of shots; make sure they're exactly balanced
    sampler = FewShotSampler(dataset, 6, rng=Random(42))
    for batch in islice(sampler, 5):
        counter = Counter(sample[label_col] for sample in batch)

        # Check if the output batch is balanced
        assert counter[0] == counter[1]

    # Start with an odd number of shots; make sure they're roughly balanced
    sampler = FewShotSampler(dataset, 5, rng=Random(42))
    for batch in islice(sampler, 5):
        counter = Counter(sample[label_col] for sample in batch)

        # The batch should be balanced to within 1 sample
        assert abs(counter[0] - counter[1]) <= 1


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
    for sample in islice(reservoir, 3000):
        counter[sample[col]] += 1

    # Check if the output is roughly balanced
    label_0_count = counter[0]
    label_1_count = counter[1]
    imbalance = abs(label_0_count - label_1_count) / (label_0_count + label_1_count)

    # Set a tolerance threshold for the imbalance ratio (e.g., 1%)
    tol = 0.01
    assert imbalance < tol, f"Imbalance ratio {imbalance} exceeded tolerance {tol}"
