from collections import deque
from dataclasses import InitVar, dataclass, field
from itertools import cycle
from random import Random
from typing import Hashable, Iterable, Iterator, Optional

from datasets import Features, IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from ..utils import infer_label_column
from ..utils.math_util import stochastic_round_constrained
from ..utils.typing import assert_type


@dataclass
class BalancedSampler(TorchIterableDataset):
    """
    A sampler that approximately balances a multi-class classification dataset in a
    streaming fashion."""

    data: Iterable[dict]
    """The input dataset to balance."""
    label_choices: InitVar[set[Hashable]]
    """The set of all possible class labels."""
    buffer_size: int = 1000
    """The per-class buffer size to use for balancing the dataset."""
    buffers: dict[Hashable, deque[dict]] = field(default_factory=dict, init=False)
    """The buffers used for balancing the dataset."""
    label_col: str = "label"
    """The name of the column containing the class labels."""

    def __post_init__(self, label_choices: set[Hashable]):
        # Initialize empty buffers
        self.buffers = {
            label: deque(maxlen=self.buffer_size) for label in label_choices
        }

    def __iter__(self):
        for sample in self.data:
            label = sample[self.label_col]
            if label not in self.buffers:
                raise ValueError(
                    f"Expected label to be one of {self.buffers}, got {label}"
                )

            # Add the sample to the buffer for its class label
            self.buffers[label].append(sample)

            # Check if all buffers have at least one sample
            while all(len(buffer) > 0 for buffer in self.buffers.values()):
                # Yield one sample from each buffer in a round-robin fashion
                for buf in self.buffers.values():
                    yield buf.popleft()


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
        feats = assert_type(Features, dataset.features)
        self.label_col = label_col or infer_label_column(feats)
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
