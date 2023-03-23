from ..utils import infer_label_column
from ..math_util import stochastic_round_constrained
from dataclasses import dataclass, field, InitVar
from datasets import IterableDataset
from itertools import cycle
from random import Random
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
    seed: InitVar[int] = 42

    def __post_init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

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


class FewShotSampler:
    """Yields batches of few-shot examples that are as balanced as possible.

    If the number of examples is divisible by the number of shots, this sampler
    will yield batches of exactly `num_shots` examples. Otherwise, it will
    use `stochastic_round_constrained` to get as close to balanced batches as
    possible.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        num_shots: int,
        rng: Random,
        label_col: Optional[str] = None,
    ):
        self.dataset = dataset
        self.label_col = label_col or infer_label_column(dataset.features)
        self.num_shots = num_shots
        self.rng = rng

    def __iter__(self) -> Iterator[list[dict]]:
        neg_buf, pos_buf = [], []

        # Infinite loop over the dataset!
        for sample in cycle(self.dataset):
            label = sample[self.label_col]
            if label == 0:
                neg_buf.append(sample)
            elif label == 1:
                pos_buf.append(sample)
            else:
                raise ValueError(f"Expected label to be 0 or 1, got {label}")

            neg_count, pos_count = stochastic_round_constrained(
                [self.num_shots / 2, self.num_shots / 2], self.rng
            )
            while len(neg_buf) >= neg_count and len(pos_buf) >= pos_count:
                batch = []
                for _ in range(neg_count):
                    batch.append(neg_buf.pop())
                for _ in range(pos_count):
                    batch.append(pos_buf.pop())

                self.rng.shuffle(batch)
                yield batch
