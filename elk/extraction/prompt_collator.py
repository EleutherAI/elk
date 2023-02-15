from dataclasses import dataclass
from datasets import DatasetDict, load_dataset  # type: ignore
from promptsource.templates import DatasetTemplates
from random import Random
from typing import Literal, Optional
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist

from elk.extraction.dataset_preprocessing import undersample


@dataclass
class Prompt:
    """A prompt is a question, its possible answers, and a label."""

    question: str
    answers: list[str]
    label: int

    def to_string(self, answer_idx: int, sep: str = "\n") -> str:
        return f"{self.question}{sep}{self.answers[answer_idx]}"


class PromptCollator(Dataset):
    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: str,
        label_column: str = "label",
        max_examples: int = 0,
        seed: int = 42,
        strategy: Literal["all", "randomize"] = "randomize",
        balance: bool = False,
    ):
        data = load_dataset(path, name)
        assert isinstance(data, DatasetDict)

        # Create a train-test split if needed
        train_name, *others = data.keys()
        if not others:
            print("Creating a train-test split...")
            data = data[train_name].train_test_split(
                seed=seed, shuffle=False, stratify_by_column=label_column
            )

        if split not in data and split == "validation":
            print("No validation split found, using test split instead")
            split = "test"

        self.dataset = data[split]

        if balance:
            self.dataset = undersample(self.dataset, seed, label_column)

        self.labels, counts = np.unique(self.dataset[label_column], return_counts=True)
        self.label_fracs = counts / counts.sum()

        print(f"Class balance '{split}': {[f'{x:.2%}' for x in self.label_fracs]}")
        pivot, *rest = self.label_fracs
        if not all(x == pivot for x in rest):
            print("Use arg --balance to force class balance")

        if len(self.labels) < 2:
            raise ValueError(f"Dataset {path}/{name} has only one label")
        if dist.is_initialized():
            self.dataset = self.dataset.shard(dist.get_world_size(), dist.get_rank())

        self.dataset = self.dataset.shuffle(seed=seed)
        if max_examples:
            num_examples = len(self.dataset)
            if max_examples > num_examples:
                max_examples = num_examples
            self.dataset = self.dataset.select(range(max_examples))

        self.label_column = label_column
        self.prompter = DatasetTemplates(path, subset_name=name)  # type: ignore
        self.rng = Random(seed)
        self.strategy = strategy

    def __getitem__(self, index: int) -> Prompt:
        prompts = list(self.prompter.templates.values())

        if self.strategy == "all":
            example_idx, prompt_idx = divmod(index, len(prompts))
            example = self.dataset[example_idx]
            template = prompts[prompt_idx]

        elif self.strategy == "randomize":
            example = self.dataset[index]
            template = self.rng.choice(prompts)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        true_label = example[self.label_column]
        answers = []
        questions = set()

        for fake_label in self.labels:
            example[self.label_column] = fake_label

            q, a = template.apply(example)
            answers.append(a)
            questions.add(q)

        assert len(questions) == 1
        return Prompt(question=questions.pop(), answers=answers, label=true_label)

    def __iter__(self):
        return (self[i] for i in range(len(self.dataset)))

    def __len__(self):
        N = len(self.dataset)
        if self.strategy == "all":
            N *= len(self.prompter.templates)

        return N
