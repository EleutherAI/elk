"""Collator for prompts."""

from ..math import stochastic_round_constrained
from dataclasses import dataclass
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from numpy.typing import NDArray
from promptsource.templates import DatasetTemplates
from random import Random
from torch.utils.data import Dataset as TorchDataset
from typing import Literal, Optional, cast
import numpy as np
import torch.distributed as dist


@dataclass
class Prompt:
    """A question, its possible answers, and the correct answer."""

    question: str
    answers: list[str]
    label: int

    def to_string(self, answer_idx: int, sep: str = "\n") -> str:
        return f"{self.question}{sep}{self.answers[answer_idx]}"


class PromptCollator(TorchDataset):
    """Wrapper for a HuggingFace dataset which generates prompts with `promptsource`.

    Usually `promptsource` has multiple prompt templates for a given dataset. We handle
    this in two ways. When `strategy` is set to `"randomize"` (the default), we sample
    a random prompt template for each example on-the-fly when `__getitem__` is called,
    using the seed passed to `__init__`. Note this means that the same example may be
    assigned a different prompt template if `__getitem__` is called multiple times with
    the same index.

    When `strategy` is set to `"all"`, we "broadcast" the prompt templates across the
    dataset, multiplying its effective size by the number of templates.

    Example:
    >>> collator = PromptCollator("super_glue", "boolq", split="train")
    >>> prompt = collator[0]
    >>> prompt.to_string(0)
    "Henry Mills (Once Upon a Time) -- Henry Daniel Mills is a fictional character...

    Args:
        path: First part of the dataset path, e.g. `"imdb"`.
        name: Optional second part of the dataset path, e.g. `"boolq"`.
        split: The split to use.
        balance: Whether to force class balance in the dataset using undersampling.
        label_column: The column containing the labels. By default, we infer this from
            the datatypes of the columns in the dataset; if there is only one column
            with a `ClassLabel` datatype, we use that.
        max_examples: The maximum number of examples to use from the dataset. If zero,
            use all examples. Defaults to 0.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot. Defaults to 0.
        seed: The seed to use for prompt randomization. Defaults to 42.
        strategy: The strategy to use for assigning prompt templates to examples. See
            above for details. Defaults to `"randomize"`.
    """

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: str,
        balance: bool = False,
        label_column: Optional[str] = None,
        max_examples: int = 0,
        num_shots: int = 0,
        seed: int = 42,
        strategy: Literal["all", "randomize"] = "randomize",
    ):
        self.num_shots = num_shots
        self.prompter = DatasetTemplates(path, subset_name=name)  # type: ignore
        self.rng = Random(seed)
        self.strategy = strategy

        # TODO: Should we support IterableDataset?
        ds_dict = load_dataset(path, name)
        assert isinstance(ds_dict, DatasetDict)

        # By default we use the existing train-validation/test split in the dataset.
        # If it doesn't exist, we create our own 75/25 train-test split. Crucially,
        # because the RNG is always seeded, this split will be the same for independent
        # instantiations of PromptCollator (unless you set the seed to something else).
        # This allows you to just set split="train" and split="test" for any dataset
        # and not worry about train-test leakage.
        split_name, *others = ds_dict.keys()
        if not others:
            print("Creating a 75/25 train-test split...")

            # Don't shuffle now because we're going to shuffle later
            ds_dict = ds_dict[split_name].train_test_split(
                seed=seed, shuffle=False, stratify_by_column=label_column
            )

        # Lots of datasets have a validation split or a test split, but not both. If
        # the requested split doesn't exist, we try to use the other one instead.
        if split not in ds_dict and split in ("validation", "test"):
            new_split = "test" if split == "validation" else "train"
            if new_split in ds_dict:
                print(f"No {split} split found, using {new_split} split instead")
                split = new_split
            else:
                raise ValueError(f"No {split} or {new_split} split found")

        # The 'active' split is the one that gets queried by __getitem__ in the
        # zero-shot case.
        self.active_split = ds_dict[split]
        features = self.active_split.features

        # We infer that the label is the unique ClassLabel column in the dataset
        if label_column is None:
            label_cols = [
                col for col, dtype in features.items() if isinstance(dtype, ClassLabel)
            ]
            if not label_cols:
                raise ValueError(f"Dataset {path} has no label column")
            elif len(label_cols) > 1:
                raise ValueError(
                    f"Dataset {path} has multiple label columns {label_cols}; specify "
                    "--label-column to disambiguate"
                )
            else:
                label_column = label_cols[0]
                assert isinstance(label_column, str)

                print(f"Using label column '{label_column}'")

        # Make sure manually specified label columns exist and are ClassLabels.
        # NOTE: Should we actually support non-ClassLabel labels? The essential thing
        # is that we know the number of classes, and ClassLabel makes that easy.
        elif label_column not in features:
            raise ValueError(f"Dataset {path} has no column '{label_column}'")
        elif not isinstance(features[label_column], ClassLabel):
            raise ValueError(
                f"Column '{label_column}' in dataset {path} is not a `ClassLabel`"
            )

        # Sanity check the label column
        self.label_column = label_column
        if self.num_classes < 2:
            raise ValueError(f"Dataset {path} should have more than one class")

        # Compute the empirical class balance in the active split. Sanity check that
        # all class mentioned in the ClassLabel datatype are represented.
        class_sizes = np.bincount(
            self.active_split[label_column], minlength=self.num_classes
        )
        if not np.all(class_sizes > 0):
            missing = np.flatnonzero(class_sizes == 0).tolist()
            raise ValueError(f"Dataset {path} has missing classes: {missing}")

        # Enforce class balance if needed
        if balance:
            smallest_size = class_sizes.min()
            print(f"Undersampling classes to {smallest_size} examples each")

            # First group the active split by class
            strata = [
                self.active_split.filter(lambda ex: ex[label_column] == i)
                for i in range(self.num_classes)
            ]
            # Then randomly sample `smallest_size` examples from each class and merge
            self.active_split = cast(
                Dataset,
                concatenate_datasets(
                    [
                        stratum.select(
                            self.rng.sample(range(len(stratum)), k=smallest_size)
                        )
                        for stratum in strata
                    ]
                ),
            )

            # Sanity check that we successfully balanced the classes
            class_sizes = np.bincount(
                self.active_split[label_column], minlength=self.num_classes
            )
            assert np.all(class_sizes == smallest_size)

        # Store the (possibly post-undersampling) empirical class balance for later
        self.class_fracs: NDArray[np.floating] = class_sizes / class_sizes.sum()

        # Shard across ranks iff we're in a distributed setting
        if dist.is_initialized():
            self.active_split = self.active_split.shard(
                dist.get_world_size(), dist.get_rank()
            )

        # We use stratified sampling to create few-shot prompts that are as balanced as
        # possible. If needed, create the strata now so that we can use them later.
        if num_shots > 0:
            # Sanity check that we can fit an example from every class in the prompt
            if self.num_classes > num_shots:
                raise ValueError(
                    f"Few-shot prompts should contain at least one example from each "
                    f"class; got {num_shots} examples and {self.num_classes} classes"
                )

            # Sanity check to prevent train-test leakage via few-shot prompts
            if "train" not in ds_dict:
                raise ValueError(
                    f"Dataset {path} has no train split, so we can't create "
                    "few-shot prompts"
                )

            self.fewshot_strata = [
                ds_dict["train"].filter(lambda ex: ex[label_column] == i)
                for i in range(self.num_classes)
            ]
        else:
            self.fewshot_strata: list[Dataset] = []

        # Now shuffle the active split and truncate it if needed
        self.active_split = self.active_split.shuffle(seed=seed)
        if 0 < max_examples < len(self.active_split):
            self.active_split = self.active_split.select(range(max_examples))

    def __getitem__(self, index: int) -> Prompt:
        prompts = list(self.prompter.templates.values())

        if self.strategy == "all":
            example_idx, prompt_idx = divmod(index, len(prompts))
            example = self.active_split[example_idx]
            template = prompts[prompt_idx]

        elif self.strategy == "randomize":
            example = self.active_split[index]
            template = self.rng.choice(prompts)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        true_label = example[self.label_column]
        answers = []
        questions = set()

        for fake_label in range(self.num_classes):
            example[self.label_column] = fake_label

            q, a = template.apply(example)
            answers.append(a)
            questions.add(q)

        assert len(questions) == 1
        question = questions.pop()

        if self.num_shots > 0:
            # Use stratified sampling to get `num_shots` examples from the train set.
            # If `num_shots` is not divisible by the number of classes, stochastic
            # rounding is used to determine the number of examples per class.
            example_counts = stochastic_round_constrained(
                list(self.class_fracs * self.num_shots), self.rng
            )
            examples = []

            for count, stratum in zip(example_counts, self.fewshot_strata):
                indices = self.rng.sample(range(len(stratum)), count)

                for idx in indices:
                    q, a = template.apply(stratum[idx])
                    examples.append(f"{q}\n{a}")

            self.rng.shuffle(examples)
            question = "\n\n".join(examples + [question])

        return Prompt(question=question, answers=answers, label=true_label)

    def __iter__(self):
        return (self[i] for i in range(len(self.active_split)))

    def __len__(self):
        """Get the number of prompts in the dataset."""
        N = len(self.active_split)
        if self.strategy == "all":
            N *= len(self.prompter.templates)

        return N

    @property
    def num_classes(self) -> int:
        """Number of classes in the underlying dataset."""

        # We piggyback on the ClassLabel feature type to get the number of classes
        return self.active_split.features[self.label_column].num_classes
