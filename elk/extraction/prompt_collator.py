from dataclasses import dataclass
from datasets import Sequence, DatasetDict, Features, Value, load_dataset
from promptsource.templates import DatasetTemplates
from random import Random
from typing import Literal, Optional
import numpy as np


@dataclass
class Prompt:
    """A prompt is a question, its possible answers, and a label."""

    question: str
    answers: list[str]
    label: int

    def to_string(self, answer_idx: int, sep: str = "\n") -> str:
        return f"{self.question}{sep}{self.answers[answer_idx]}"


class PromptCollator:
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
        """Create a prompt collator.
        @param strategy: "all" will generate all prompts for all examples, while
            "randomize" will generate a random prompt for each example."""
        data = load_dataset(path, name)
        assert isinstance(data, DatasetDict)

        # Create a train-test split if needed
        train_name, *others = data.keys()
        if not others:
            print("Creating a train-test split...")
            data = data[train_name].train_test_split(
                seed=seed, stratify_by_column=label_column
            )
        else:
            data = data.shuffle(seed)

        if split not in data and split == "validation":
            print("No validation split found, using test split instead")
            split = "test"

        self.dataset = data[split]
        self.labels, counts = np.unique(self.dataset[label_column], return_counts=True)
        self.label_fracs = counts / counts.sum()
        if len(self.labels) < 2:
            raise ValueError(f"Dataset {path}/{name} has only one label")
        if max_examples:
            self.dataset = self.dataset.select(range(max_examples))

        self.label_column = label_column
        self.prompter = DatasetTemplates(path, subset_name=name)  # type: ignore
        self.rng = Random(seed)
        self.strategy = strategy
        self._apply_prompts()

    def _apply_batch(self, examples: dict) -> dict:
        """@param examples: a dict of lists of examples"""
        batch_size = len(examples[self.label_column])

        out_examples = {
            "template_name": [],
            "predicate": [],
            "answers": [],
            "label": [],
        }
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            if self.strategy == "all":
                templates = self.prompter.templates.items()
            elif self.strategy == "randomize":
                templates = [self.rng.choice(list(self.prompter.templates.items()))]
            else:
                raise ValueError(f"Unknown strategy {self.strategy}")
            for template_name, template in templates:
                true_label = example[self.label_column]
                answers = []
                questions = set()
                for fake_label in self.labels:
                    example[self.label_column] = fake_label

                    q, a = template.apply(example)
                    answers.append(a)
                    questions.add(q)

                assert len(questions) == 1
                prompt = Prompt(
                    question=questions.pop(), answers=answers, label=true_label
                )
                out_examples["template_name"].append(template_name)
                out_examples["predicate"].append(prompt.question)
                out_examples["answers"].append(prompt.answers)
                out_examples["label"].append(prompt.label)

        return out_examples

    def _apply_prompts(self):
        """Modify the dataset"""
        self.dataset = self.dataset.map(
            self._apply_batch,
            batched=True,
            batch_size=100,
            features=Features(
                {
                    "template_name": Value("string"),
                    "predicate": Value("string"),
                    "answers": Sequence(
                        feature=Value("string"), length=len(self.labels)
                    ),
                    "label": Value("int32"),
                    "text": Value("string"),
                }
            ),
        )

    def __getitem__(self, index: int) -> dict:
        return self.dataset[index]

    def __iter__(self):
        return (self[i] for i in range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
