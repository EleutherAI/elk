from ..math import stochastic_round_constrained
from ..promptsource import DatasetTemplates
from ..utils import assert_type, compute_class_balance, infer_label_column, undersample
from dataclasses import dataclass
from datasets import DatasetDict, load_dataset
from numpy.typing import NDArray
from random import Random
from simple_parsing.helpers import field, Serializable
from torch.utils.data import Dataset as TorchDataset
from typing import Optional
import numpy as np


@dataclass
class Prompt:
    """A question, its possible answers, and the correct answer.
    A question is a template applied to a predicate."""

    question: str
    answers: list[str]
    label: int
    template_name: str

    def to_string(self, answer_idx: int, sep: str = "\n") -> str:
        return f"{self.question}{sep}{self.answers[answer_idx]}"


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
        max_examples: The maximum number of examples to use from the dataset. If zero,
            use all examples. Defaults to 0.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot. Defaults to 0.
        seed: The seed to use for prompt randomization. Defaults to 42.
        num_variants: The number of prompt templates to apply to each predicate upon
            call to __getitem__. Use -1 to apply all available templates. Defaults to 1.
    """

    dataset: str = field(positional=True)
    balance: bool = False
    label_column: Optional[str] = None
    max_examples: int = 0
    num_shots: int = 0
    seed: int = 42
    num_variants: int = 1


class PromptDataset(TorchDataset):
    """Wrapper for a HuggingFace dataset which generates prompts with `promptsource`.

    Usually `promptsource` has multiple prompt templates for a given dataset. We handle
    this in two ways. When `strategy` is set to `"randomize"` (the default), we sample
    a random prompt template for each example on-the-fly when `__getitem__` is called,
    using the seed passed to `__init__`. Note this means that the same example may be
    assigned a different prompt template if `__getitem__` is called multiple times with
    the same index.  TODO redo this documentation

    When `strategy` is set to `"all"`, we "broadcast" the prompt templates across the
    dataset, multiplying its effective size by the number of templates.

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

        # Enforce class balance if needed
        if cfg.balance:
            self.active_split = undersample(self.active_split, self.rng, label_col)
            self.class_fracs = np.ones(self.num_classes) / self.num_classes
        else:
            class_sizes = compute_class_balance(self.active_split, label_col)
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

            # Sanity check to prevent train-test leakage via few-shot prompts
            if "train" not in ds_dict:
                raise ValueError(
                    f"Dataset {cfg.dataset} has no train split, so we can't create "
                    "few-shot prompts"
                )

            self.fewshot_strata = [
                ds_dict["train"].filter(lambda ex: ex[label_col] == i)
                for i in range(self.num_classes)
            ]
        else:
            self.fewshot_strata = []

        # Now shuffle the active split and truncate it if needed
        self.active_split = self.active_split.shuffle(seed=cfg.seed)
        if 0 < cfg.max_examples < len(self.active_split):
            self.active_split = self.active_split.select(range(cfg.max_examples))

        # Shard if needed
        if world_size > 1:
            self.active_split = self.active_split.shard(world_size, rank)

    def __getitem__(self, index: int) -> list[Prompt]:
        """Get a list of prompts for a given predicate"""
        # get self.num_variants unique prompts from the template pool
        template_names = self.rng.sample(
            self.prompter.templates.keys(), self.num_variants
        )

        example = self.active_split[index]

        prompts = []
        for template_name in template_names:
            template = self.prompter.templates[template_name]

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
                        q, a = template.apply(stratum[idx])
                        examples.append(f"{q}\n{a}")

                self.rng.shuffle(examples)
                question = "\n\n".join(examples + [question])

            prompts.append(
                Prompt(
                    question=question,
                    answers=answers,
                    label=true_label,
                    template_name=template_name,
                )
            )
        return prompts

    def __iter__(self):
        return (self[i] for i in range(len(self.active_split)))

    def __len__(self):
        """Get the number of predicates in the dataset."""
        return len(self.active_split)

    @property
    def num_classes(self) -> int:
        """Number of classes in the underlying dataset."""

        # We piggyback on the ClassLabel feature type to get the number of classes
        return self.active_split.features[self.label_column].num_classes
