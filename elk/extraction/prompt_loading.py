from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from os.path import exists
from random import Random
from typing import Any, Iterator, Literal, Optional

from datasets import (
    Dataset,
    Features,
    load_dataset,
    load_dataset_builder,
)
from datasets.distributed import split_dataset_by_node
from simple_parsing.helpers import Serializable, field

from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    infer_label_column,
    infer_num_classes,
    select_train_val_splits,
)
from .balanced_sampler import BalancedSampler, FewShotSampler


@dataclass
class PromptConfig(Serializable):
    """
    Args:
        datasets: List of space-delimited names of the HuggingFace datasets to use, e.g.
            [`"super_glue boolq", "imdb"]`.
        balance: Whether to force class balance in the dataset using undersampling.
        data_dir: The directory to use for caching the dataset. Defaults to
            `~/.cache/huggingface/datasets`.
        label_column: The column containing the labels. By default, we infer this from
            the datatypes of the columns in the dataset; if there is only one column
            with a `ClassLabel` datatype, we use that.
        max_examples: The maximum number of examples to use from the val dataset.
            If a single number, use at most that many examples for each split. If a list
            of length 2, use the first element for the train split and the second for
            the val split. If empty, use all examples. Defaults to empty.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot. Defaults to 0.
        num_variants: The number of prompt templates to apply to each predicate upon
            call to __getitem__. Use -1 to apply all available templates. Defaults to
            -1.
        seed: The seed to use for prompt randomization. Defaults to 42.
        stream: Whether to stream the dataset from the Internet. Defaults to False.
        combined_template_output_path: Path to save a combined template file to, when
            applying prompt invariance across multiple datasets. Interpreted as a
            subpath of `combined_paths` in the templates dir. Defaults to empty string.
    """

    datasets: list[str] = field(positional=True)
    data_dirs: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    max_examples: list[int] = field(default_factory=lambda: [750, 250])
    num_classes: int = 0
    num_shots: int = 0
    num_variants: int = -1
    seed: int = 42
    stream: bool = False
    combined_template_output_path: str = ""

    def __post_init__(self):
        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )
        if not self.max_examples:
            self.max_examples = [int(1e100)]

        self.combine_templates()

        # Broadcast the limit to all splits
        if len(self.max_examples) == 1:
            self.max_examples *= 2

        # Broadcast the dataset name to all data_dirs and label_columns
        if len(self.data_dirs) == 1:
            self.data_dirs *= len(self.datasets)
        elif self.data_dirs and len(self.data_dirs) != len(self.datasets):
            raise ValueError(
                "data_dirs should be a list of length 0, 1, or len(datasets),"
                f" but got {len(self.data_dirs)}"
            )

        if len(self.label_columns) == 1:
            self.label_columns *= len(self.datasets)
        elif self.label_columns and len(self.label_columns) != len(self.datasets):
            raise ValueError(
                "label_columns should be a list of length 0, 1, or len(datasets),"
                f" but got {len(self.label_columns)}"
            )

    def combine_templates(self):
        if not self.combined_template_output_path:
            return

        print(
            "Copying templates across datasets to combined_templates/ "
            + f"{self.combined_template_output_path}/templates.yaml"
        )
        combined_prompter = DatasetTemplates(
            "combined_templates", self.combined_template_output_path
        )
        combined_prompter.templates = {}
        ref_ds_builder = None
        for i, ds_string in enumerate(self.datasets):
            ds_name, _, config_name = ds_string.partition(" ")
            ds_builder = load_dataset_builder(ds_name, config_name or None)
            if i == 0:
                # Set first dataset as reference
                ref_ds_builder = ds_builder
            elif not self.verify_cols(ds_builder, ref_ds_builder):
                return

            # Once verified, merge templates.
            prompter = DatasetTemplates(ds_name, config_name)
            combined_prompter.merge_templates_from(prompter)
        print("Total number of templates: ", len(combined_prompter.templates))
        combined_prompter.write_to_file()
        print(
            "Saved to promptsource/templates/combined_templates/"
            + f"{self.combined_template_output_path}.yaml"
        )

    def verify_cols(self, ds_builder, ref_ds_builder) -> bool:
        """Verify that number of features and number of classes for ClassLabel
        match the expected values.
        """
        expected_features = len(ref_ds_builder.info.features)
        expected_classes = ref_ds_builder.info.features["label"].num_classes
        num_features = len(ds_builder.info.features)
        num_classes = ds_builder.info.features["label"].num_classes
        if expected_features > 0 and num_features != expected_features:
            print(
                "WARNING: Datasets do not have the same number of features;",
                f"{ds_name} has {num_features} features while first dataset has",
                f"{expected_features}. Prompting datasets separately.",
            )
            return False
        if expected_classes > 0 and num_classes != expected_classes:
            print(
                "WARNING: Datasets do not have the same number of ClassLabel classes",
                f"{ds_name} has {num_classes} classes while first dataset has",
                f"{expected_classes}. Prompting datasets separately.",
            )
            return False
        return True

    def explode(self) -> list["PromptConfig"]:
        """Explode the config into a list of configs, one for each dataset."""
        copies = []

        for ds, data_dir, col in zip_longest(
            self.datasets, self.data_dirs, self.label_columns
        ):
            copy = deepcopy(self)
            copy.datasets = [ds]
            copy.data_dirs = [data_dir] if data_dir else []
            copy.label_columns = [col] if col else []
            copies.append(copy)

        return copies


def load_prompts(
    ds_string: str,
    label_column: Optional[str] = None,
    num_classes: int = 0,
    num_shots: int = 0,
    num_variants: int = -1,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    stream: bool = False,
    rank: int = 0,
    world_size: int = 1,
    combined_template_output_path: str = "",
) -> Iterator[dict]:
    """Load a dataset full of prompts generated from the specified dataset.

    Args:
        ds_string: Space-delimited name of the HuggingFace dataset to use,
            e.g. `"super_glue boolq"` or `"imdb"`.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        stream: Whether to stream the dataset from the Internet. Defaults to False.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(" ")

    prompter = None
    if combined_template_output_path and exists(combined_template_output_path):
        prompter = DatasetTemplates("combined_templates", combined_template_output_path)
    else:
        prompter = DatasetTemplates(ds_name, config_name)

    ds_dict = assert_type(
        dict, load_dataset(ds_name, config_name or None, streaming=stream)
    )
    train_name, val_name = select_train_val_splits(ds_dict)
    split_name = val_name if split_type == "val" else train_name

    ds = ds_dict[split_name].shuffle(seed=seed)
    train_ds = ds_dict[train_name].shuffle(seed=seed)
    if not stream:
        ds = assert_type(Dataset, ds)
        if world_size > 1:
            ds = ds.shard(world_size, rank)

        ds = ds.to_iterable_dataset().cast(ds.features)

    elif world_size > 1:
        # This prints to stdout which is slightly annoying
        ds = split_dataset_by_node(dataset=ds, rank=rank, world_size=world_size)

    num_templates = len(prompter.templates)
    num_variants = (
        num_templates if num_variants == -1 else min(num_variants, num_templates)
    )
    assert num_variants > 0
    if rank == 0:
        print(f"Using {num_variants} variants of each prompt")

    label_column = label_column or infer_label_column(ds.features)
    num_classes = num_classes or infer_num_classes(ds.features[label_column])
    rng = Random(seed)

    if num_shots > 0:
        fewshot = FewShotSampler(
            train_ds,  # TODO: not iterator
            num_shots=num_shots,
            rng=rng,
        )
        fewshot_iter = iter(fewshot)
    else:
        fewshot_iter = None

    # Remove everything except the label column
    extra_cols = list(assert_type(Features, ds.features))
    extra_cols.remove(label_column)

    for example in BalancedSampler(ds, num_classes, label_col=label_column):
        yield _convert_to_prompts(
            example,
            label_column=label_column,
            num_classes=num_classes,
            num_variants=num_variants,
            prompter=prompter,
            rng=rng,
            fewshot_iter=fewshot_iter,
        )


def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    label_column: str,
    num_classes: int,
    num_variants: int,
    rng: Random,
    fewshot_iter: Optional[Iterator[list[dict]]] = None,
) -> dict[str, Any]:
    """Prompt-generating function to pass to `IterableDataset.map`."""
    labels_are_strings = isinstance(example[label_column], str)
    prompts = []
    templates = list(prompter.templates.values())
    if num_variants < len(templates):
        templates = rng.sample(templates, num_variants)

    def qa_cat(q: str, a: str) -> str:
        # if the jinja template already adds whitespace, don't add more
        sep = "" if not q or q[-1].isspace() or not a or a[0].isspace() else " "
        return f"{q}{sep}{a}" if a and not a.isspace() else q

    # For sanity checking that prompts are unique
    prompt_counter = Counter()
    label_indices = set()

    for template in templates:
        choices = []
        string_choices = template.get_answer_choices_list(example)

        label = example[label_column]
        label_indices.add(string_choices.index(label) if labels_are_strings else label)

        for answer_idx in range(num_classes):
            fake_example = example.copy()
            if labels_are_strings:
                fake_example[label_column] = string_choices[answer_idx]
            else:
                fake_example[label_column] = answer_idx

            q, a = template.apply(fake_example)
            text = qa_cat(q, a or string_choices[answer_idx])
            prompt_counter[text] += 1

            if fewshot_iter is not None:
                # Infinite iterator so we don't need to worry about StopIteration
                fewshot_examples = next(fewshot_iter)
                fewshot_texts = [
                    qa_cat(q, a) for q, a in map(template.apply, fewshot_examples)
                ]
                text = "\n\n".join(fewshot_texts) + "\n\n" + text

            choices.append(
                dict(
                    # Strip whitespace from the answer to make it easier to
                    # compare with the model's output
                    answer=a.strip(),
                    text=text,
                )
            )

        prompts.append(choices)

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    # Sanity check: label should be the same across all variants
    if len(label_indices) > 1:
        raise ValueError(
            f"Label index should be the same all variants, but got {label_indices}"
        )

    return dict(
        label=label_indices.pop(),
        prompts=prompts,
        template_names=[template.name for template in templates],
    )
