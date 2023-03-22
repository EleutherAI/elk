from ..utils import infer_label_column
from collections import Counter
from dataclasses import dataclass, field, InitVar
from datasets import IterableDataset
from itertools import cycle
from torch.utils.data import IterableDataset as TorchIterableDataset
from typing import Iterator, Optional
import numpy as np


@dataclass
class BalancedSampler(TorchIterableDataset):
    """
    Approximately balances a binary classification dataset in a streaming fashion.
    Written mostly by GPT-4.

    Args:
        dataset (IterableDataset): The HuggingFace IterableDataset to balance.
        label_col (Optional[str], optional): The name of the column containing the
            binary label. If not provided, the label column will be inferred from
            the dataset features. Defaults to None.
        buffer_size (int, optional): The total buffer size to use for balancing the
            dataset. This value should be divisible by 2, as it will be equally
            divided between the two binary label values (0 and 1). Defaults to 1000.
    """

    dataset: IterableDataset
    label_counts: np.ndarray = field(default_factory=lambda: np.zeros(2))
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def __iter__(self):
        for sample in self.dataset:
            label = sample["label"]

            # Update class counts
            self.label_counts[label] += 1
            current_balance = self.label_counts / self.label_counts.sum()

            # Check if the sample should be dropped
            majority_class = np.argmax(current_balance)
            if label == majority_class:
                # Solution of n * p * q / [n * (1 - p) + n * p * q] = 0.5 for q
                keep_prob = 1 / current_balance[majority_class] - 1
                if self.rng.uniform() < 1 - keep_prob:
                    continue

            yield sample


class BalancedBatchSampler:
    """Yields precisely balanced batches from a binary classification dataset.

    Written by a human being because GPT-4 couldn't figure out how to do it.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        label_col: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.label_col = label_col or infer_label_column(dataset.features)

    def __iter__(self) -> Iterator[list[dict]]:
        batch = []

        max_count = self.batch_size // 2
        label_counts = Counter()

        # Infinite loop!
        for sample in cycle(self.dataset):
            label = sample[self.label_col]
            if label_counts[label] >= max_count:
                continue

            batch.append(sample)
            label_counts[label] += 1

            if len(batch) == self.batch_size:
                yield batch

                batch = []
                label_counts.clear()
