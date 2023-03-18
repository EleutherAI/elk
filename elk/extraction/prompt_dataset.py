from ..math_util import stochastic_round_constrained
from ..promptsource import DatasetTemplates, Template
from ..utils import (
    apply_template,
    assert_type,
    binarize,
    compute_class_balance,
    infer_label_column,
    infer_num_classes,
    select_train_val_splits,
    undersample,
)
from dataclasses import dataclass
from datasets import DatasetDict, load_dataset, ClassLabel, Value
from numpy.typing import NDArray
from random import Random
from simple_parsing.helpers import field, Serializable
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Any
import numpy as np


@dataclass
class Prompt:
    """A prompt for a single example in a dataset"""

    prefix: str
    template: Template
    example: dict[str, Any]
    label: int
    label_column: str

    def to_string(self, answer_idx: int) -> str:
        """Return the prompt as a string, with the answer at `answer_idx`."""
        fake_example = self.example.copy()
        fake_example[self.label_column] = answer_idx
        return self.prefix + apply_template(self.template, fake_example)


@dataclass
class PromptConfig(Serializable):
    """
    Args:
        dataset: Space-delimited name of the HuggingFace dataset to use, e.g.
            `"super_glue boolq"` or `"imdb"`.
        balance: Whether to force class balance in the dataset using undersampling.
        label_column: The column containing the labels. By default, we infer this from
            the datatypes of the columns in the dataset; if there is only one column
            with a `ClassLabel` datatype, we use that.
        num_classes: The number of classes in the dataset. By default, we infer this
            from the datatypes of the columns in the dataset; if there is only one
            column with a `ClassLabel` datatype, we use the number of classes in that
            column.
        max_examples: The maximum number of examples to use from the val dataset.
            If a single number, use at most that many examples for each split. If a list
            of length 2, use the first element for the train split and the second for
            the val split. If empty, use all examples. Defaults to empty.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot. Defaults to 0.
        seed: The seed to use for prompt randomization. Defaults to 42.
        num_variants: The number of prompt templates to apply to each predicate upon
            call to __getitem__. Use -1 to apply all available templates. Defaults to 1.
    """

    dataset: str = field(positional=True)
    balance: bool = False
    label_column: Optional[str] = None
    num_classes: Optional[int] = None
    max_examples: list[int] = field(default_factory=list)
    num_shots: int = 0
    seed: int = 42
    num_variants: int = 1

    def __post_init__(self):
        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )


class PromptDataset(TorchDataset):
    """Wrapper for a HuggingFace dataset which generates prompts with `promptsource`.

    Usually `promptsource` has multiple prompt templates for a given dataset. We sample
    `num_variants` of these templates and apply them to each example in the dataset, up
    to `max_examples` examples. If `num_shots` is greater than zero, we sample that
    many examples from the dataset and use them to generate a prefix for the prompt.

    Example:
    >>> prompts = PromptDataset("super_glue", "boolq", split="train")
    >>> prompt = prompts[0]
    >>> prompt.to_string(0)
    "Henry Mills (Once Upon a Time) -- Henry Daniel Mills is a fictional character...
    """

    def __init__(
        self,
        cfg: PromptConfig,
        rank: int = 0,
        world_size: int = 1,
        split: str = "validation",
    ):
        ds_name, _, config_name = cfg.dataset.partition(" ")

        self.num_shots = cfg.num_shots
        self.prompter = DatasetTemplates(ds_name, config_name or None)  # type: ignore
        self.rng = Random(cfg.seed)
        self.num_variants = (
            cfg.num_variants if cfg.num_variants > 0 else len(self.prompter.templates)
        )

        ds_dict = assert_type(
            DatasetDict,  # TODO: Should we support IterableDataset?
            load_dataset(ds_name, config_name or None),
        )

        # By default we use the existing train-validation/test split in the dataset.
        # If it doesn't exist, we create our own 75/25 train-test split. Crucially,
        # because the RNG is always seeded, this split will be the same for independent
        # instantiations of PromptDataset (unless you set the seed to something else).
        # This allows you to just set split="train" and split="test" for any dataset
        # and not worry about train-test leakage.
        split_name, *others = ds_dict.keys()
        if not others:
            print("Creating a 75/25 train-test split...")

            # Don't shuffle now because we're going to shuffle later
            ds_dict = ds_dict[split_name].train_test_split(
                seed=cfg.seed, shuffle=False, stratify_by_column=cfg.label_column
            )
            assert isinstance(ds_dict, DatasetDict)

        # The 'active' split is the one that gets queried by __getitem__
        self.active_split = ds_dict[split]
        label_col = cfg.label_column or infer_label_column(self.active_split.features)
        self.label_column = label_col
        self.num_classes = cfg.num_classes or infer_num_classes(
            self.active_split.features[label_col]
        )

        # Enforce class balance if needed
        if cfg.balance:
            self.active_split = undersample(
                self.active_split, self.rng, self.num_classes, label_col
            )
            self.class_fracs = np.ones(self.num_classes) / self.num_classes
        else:
            class_sizes = compute_class_balance(
                self.active_split, self.num_classes, label_col
            )
            self.class_fracs: NDArray[np.floating] = class_sizes / class_sizes.sum()

        # We use stratified sampling to create few-shot prompts that are as balanced as
        # possible. If needed, create the strata now so that we can use them later.
        if cfg.num_shots > 0:
            # Sanity check that we can fit an example from every class in the prompt
            if self.num_classes > cfg.num_shots:
                raise ValueError(
                    f"Few-shot prompts should contain at least one example from each "
                    f"class; got {cfg.num_shots} examples, {self.num_classes} classes"
                )

            train_split = select_train_val_splits(ds_dict)[0]

            self.fewshot_strata = [
                ds_dict[train_split].filter(lambda ex: ex[label_col] == i)
                for i in range(self.num_classes)
            ]
        else:
            self.fewshot_strata = []

        # Now shuffle the active split and truncate it if needed
        self.active_split = self.active_split.shuffle(seed=cfg.seed)

        if cfg.max_examples:
            max_examples = (
                cfg.max_examples[0]
                if split == "train" or len(cfg.max_examples) == 1
                else cfg.max_examples[1]
            )
            if 0 < max_examples < len(self.active_split):
                self.active_split = self.active_split.select(range(max_examples))

        # Shard if needed
        if world_size > 1:
            self.active_split = self.active_split.shard(world_size, rank)

    def __getitem__(self, index: int) -> list[Prompt]:
        """Get a list of prompts for a given predicate"""
        # get self.num_variants unique prompts from the template pool
        template_names = (
            self.rng.sample(list(self.prompter.templates), self.num_variants)
            if self.num_variants < len(self.prompter.templates)
            else list(self.prompter.templates)
        )

        example = self.active_split[index]
        true_label = example[self.label_column]
        new_label = self.rng.choice([0, 1]) if self.num_classes > 2 else None

        prompts = []
        for template_name in template_names:
            template = self.prompter.templates[template_name]

            if self.num_shots > 0:
                # Use stratified sampling to get `num_shots` examples from train set.
                # If `num_shots` is not divisible by the number of classes, stochastic
                # rounding is used to determine the number of examples per class.
                example_counts = stochastic_round_constrained(
                    list(self.class_fracs * self.num_shots), self.rng
                )
                examples = []

                for count, stratum in zip(example_counts, self.fewshot_strata):
                    indices = self.rng.sample(range(len(stratum)), count)

                    for idx in indices:
                        examples.append(apply_template(template, stratum[idx]))

                self.rng.shuffle(examples)
                few_shot_prefix = "\n\n".join(examples) + "\n\n"
            else:
                few_shot_prefix = ""

            if self.num_classes > 2:
                # remove all but the true answer and one random other answer
                variant_template, variant_label = binarize(
                    template, true_label, assert_type(int, new_label), self.rng
                )
            else:
                variant_template, variant_label = template, true_label

            prompts.append(
                Prompt(
                    template=variant_template,
                    example=example,
                    label=variant_label,
                    label_column=self.label_column,
                    prefix=few_shot_prefix,
                )
            )
        return prompts

    def __iter__(self):
        return (self[i] for i in range(len(self.active_split)))

    def __len__(self):
        """Get the number of predicates in the dataset."""
        return len(self.active_split)
