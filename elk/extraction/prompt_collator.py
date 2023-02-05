from dataclasses import dataclass
from datasets import DatasetDict, load_dataset
from promptsource.templates import DatasetTemplates
from random import Random
from typing import Optional


@dataclass
class Prompt:
    """A prompt is a question and answer pair."""

    question: str
    answer: str

    def __str__(self):
        return f"{self.question}\n{self.answer}"


class PromptCollator:
    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: str,
        label_column: str = "label",
        seed: int = 42,
    ):
        data = load_dataset(path, name)
        assert isinstance(data, DatasetDict)

        # Create a train-test split if needed
        train_name, *others = data.keys()
        if not others:
            data = data[train_name].train_test_split(
                seed=seed, stratify_by_column=label_column
            )
        else:
            data = data.shuffle(seed)

        self.dataset = data[split]
        self.labels = sorted(set(self.dataset[label_column]))
        self.label_column = label_column
        self.prompter = DatasetTemplates(path, subset_name=name)  # type: ignore
        self.rng = Random(seed)

    def __getitem__(self, index: int) -> tuple[list[Prompt], int]:
        example = self.dataset[index]
        template = self.rng.choice(self.prompter.templates)
        true_label = example[self.label_column]

        prompts = []
        for fake_label in self.labels:
            example[self.label_column] = fake_label
            prompts.append(Prompt(*template.apply(example)))

        return prompts, true_label

    def __iter__(self):
        return (self[i] for i in range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
