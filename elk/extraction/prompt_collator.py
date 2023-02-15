from dataclasses import dataclass
from pathlib import Path
from datasets import DatasetDict, load_dataset  # type: ignore
from promptsource.templates import DatasetTemplates
from random import Random
from typing import Literal, Optional
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist


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
    ):
        try:
            data = load_dataset(path, name)
        except Exception:
            data_dir = str(Path(__file__).parent.parent / "datasets" / "rawdata")
            data = load_dataset(path, name, data_dir=data_dir)
        assert isinstance(data, DatasetDict)

        # Create a train-test split if needed
        train_name, *others = data.keys()
        if not others:
            print("Creating a train-test split...")
            data = data[train_name].train_test_split(
                seed=seed, shuffle=False, stratify_by_column=label_column
            )

        # For datasets of the form (validation, test)
        if "train" not in data:
            if split == "train":
                print("No train split found, using validation split instead")
                split = "validation"
            else:
                assert split == "validation"
                print("No train split found, using test split instead of validation")
                split = "test"

        # For datasets of the form (train, test)
        if split not in data and split == "validation":
            print("No validation split found, using test split instead")
            split = "test"

        self.dataset = data[split]
        self.labels, counts = np.unique(self.dataset[label_column], return_counts=True)
        self.label_fracs = counts / counts.sum()
        if len(self.labels) < 2:
            raise ValueError(f"Dataset {path}/{name} has only one label")
        if dist.is_initialized():
            self.dataset = self.dataset.shard(dist.get_world_size(), dist.get_rank())

        self.dataset = self.dataset.shuffle(seed=seed)
        if max_examples:
            num_examples = min(max_examples, len(self.dataset))
            self.dataset = self.dataset.select(range(num_examples))

        self.label_column = label_column

        self.prompters = []
        templates = DatasetTemplates(
            path,
            subset_name=name,  # type: ignore
        ).templates.values()
        for i, prompter in enumerate(templates):
            if self.is_template_valid(prompter):
                self.prompters.append(prompter)
            else:
                print(f"Template {prompter.name} is invalid")

        self.rng = Random(seed)
        self.strategy = strategy

    def is_template_valid(self, template) -> bool:
        try:
            for example in self.dataset:
                questions = set()
                for fake_label in self.labels:
                    example[self.label_column] = fake_label  # type: ignore
                    q, a = template.apply(example)
                    questions.add(q)
                if len(questions) > 1:
                    raise ValueError("Multiple questions generated")
        except Exception:
            return False
        return True

    def __getitem__(self, index: int) -> Prompt:
        if self.strategy == "all":
            example_idx, prompt_idx = divmod(index, len(self.prompters))
            example = self.dataset[example_idx]
            template = self.prompters[prompt_idx]
        elif self.strategy == "randomize":
            example = self.dataset[index]
            template = self.rng.choice(self.prompters)
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

        assert len(questions) == 1, f"Multiple questions generated: {questions}"
        return Prompt(question=questions.pop(), answers=answers, label=true_label)

    def __iter__(self):
        return (self[i] for i in range(len(self.dataset)))

    def __len__(self):
        N = len(self.dataset)
        if self.strategy == "all":
            N *= len(self.prompters)

        return N
