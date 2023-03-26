from ..math_util import stochastic_round_constrained
from ..utils import infer_label_column
from collections import deque
from dataclasses import dataclass
from datasets import IterableDataset
from itertools import cycle
from random import Random
from torch.utils.data import IterableDataset as TorchIterableDataset
from typing import Iterator, Optional, Iterable


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

    def __init__(self, data: Iterable[dict], buffer_size: int = 1000):
        self.data = data

        self.neg_buffer = deque(maxlen=buffer_size)
        self.pos_buffer = deque(maxlen=buffer_size)

    def __iter__(self):
        for sample in self.data:
            label = sample["label"]

            # Add the sample to the appropriate buffer
            if label == 0:
                self.neg_buffer.append(sample)
            else:
                self.pos_buffer.append(sample)

            while self.neg_buffer and self.pos_buffer:
                yield self.neg_buffer.popleft()
                yield self.pos_buffer.popleft()


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
