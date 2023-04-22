from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from random import Random
from typing import Any, Iterator, Literal

from datasets import ClassLabel, Dataset, Value, load_dataset
from simple_parsing.helpers import Serializable, field

from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    infer_label_column,
    select_train_val_splits,
)
from .balanced_sampler import BalancedSampler, FewShotSampler


@dataclass
class PromptConfig(Serializable):
    """
    Args:
        dataset: List of space-delimited names of the HuggingFace dataset to use, e.g.
            `"super_glue boolq"` or `"imdb"`.
        data_dir: The directory to use for caching the dataset. Defaults to
            `~/.cache/huggingface/datasets`.
        max_examples: The maximum number of examples to use from the val dataset.
            If a single number, use at most that many examples for each split. If a list
            of length 2, use the first element for the train split and the second for
            the val split. If empty, use all examples. Defaults to empty.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot. Defaults to 0.
        num_variants: The number of prompt templates to apply to each predicate upon
            call to __getitem__. Use -1 to apply all available templates. Defaults to 1.
        seed: The seed to use for prompt randomization. Defaults to 42.
    """

    datasets: list[str] = field(positional=True)
    data_dirs: list[str] = field(default_factory=list)
    max_examples: list[int] = field(default_factory=lambda: [1000, 1000])
    num_shots: int = 0
    num_variants: int = -1
    seed: int = 42

    def __post_init__(self):
        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )
        if not self.max_examples:
            self.max_examples = [int(1e100)]

        # Broadcast the limit to all splits
        if len(self.max_examples) == 1:
            self.max_examples *= 2

        # Broadcast the dataset name to all data_dirs
        if len(self.data_dirs) == 1:
            self.data_dirs *= len(self.datasets)
        elif self.data_dirs and len(self.data_dirs) != len(self.datasets):
            raise ValueError(
                "data_dirs should be a list of length 0, 1, or len(datasets),"
                f" but got {len(self.data_dirs)}"
            )

    def explode(self) -> list["PromptConfig"]:
        """Explode the config into a list of configs, one for each dataset."""
        copies = []

        for ds, data_dir in zip_longest(self.datasets, self.data_dirs):
            copy = deepcopy(self)
            copy.datasets = [ds]
            copy.data_dirs = [data_dir] if data_dir else []
            copies.append(copy)

        return copies


def load_prompts(
    ds_string: str,
    *,
    num_shots: int = 0,
    num_variants: int = -1,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[dict]:
    """Load a dataset full of prompts generated from the specified dataset.

    Args:
        ds_string: Space-delimited name of the HuggingFace dataset to use,
            e.g. `"super_glue boolq"` or `"imdb"`.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(" ")
    prompter = DatasetTemplates(ds_name, config_name)
    prompter.drop_non_mc_templates()

    ds_dict = assert_type(dict, load_dataset(ds_name, config_name or None))
    train_name, val_name = select_train_val_splits(ds_dict)
    split_name = val_name if split_type == "val" else train_name

    ds = ds_dict[split_name].shuffle(seed=seed)
    train_ds = ds_dict[train_name].shuffle(seed=seed)

    ds = assert_type(Dataset, ds)
    if world_size > 1:
        ds = ds.shard(world_size, rank)

    num_templates = len(prompter.templates)
    num_variants = (
        num_templates if num_variants == -1 else min(num_variants, num_templates)
    )
    assert num_variants > 0
    if rank == 0:
        print(f"Using {num_variants} variants of each prompt")

    # Which classes are actually present in this split of the dataset?
    # This is shockingly fast since it uses an optimized Apache Arrow primitive.
    label_column = prompter.label_column or infer_label_column(ds.features)
    observed_labels = set(ds.unique(label_column))

    # Now sanity check that the observed classes match the expected classes. This can
    # sometimes fail if we picked an unlabeled split (e.g. everything is -1)
    label_feature = ds.features[label_column]
    if isinstance(label_feature, ClassLabel):
        label_choices = {label_feature.str2int(label) for label in label_feature.names}
    elif isinstance(label_feature, Value) and label_feature.dtype == "bool":
        label_choices = {False, True}
    else:
        # We just have to assume that the observed labels are right
        label_choices = observed_labels

    if observed_labels != label_choices:
        raise ValueError(
            f"Observed labels {observed_labels} in split '{split_name}' do not match "
            f"expected labels {label_choices} from the dataset features."
        )

    if prompt_choices := prompter.label_choices:
        # The observed labels should be a superset of the prompt choices
        if not (observed_labels >= set(prompt_choices)):
            raise ValueError(
                f"Observed labels {observed_labels} in split '{split_name}' do not "
                f"match the prompt choices {prompt_choices}."
            )

        sorted_labels = prompt_choices
    else:
        # Impose a canonical order on the label choices. Theoretically the label column
        # may be of a type that doesn't support comparison (so Pylance complains), but
        # we'll just let it raise an exception if that happens.
        sorted_labels = sorted(label_choices)  # type: ignore[arg-type]

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

    ds = ds.to_iterable_dataset()
    if rank == 0:
        print(f"Label choices: {sorted_labels}")

    for example in BalancedSampler(
        ds, set(sorted_labels), label_col=label_column, strict=False
    ):
        yield _convert_to_prompts(
            example,
            label_column=label_column,
            label_choices=sorted_labels,  # type: ignore[arg-type]
            num_variants=num_variants,
            prompter=prompter,
            rng=rng,
            fewshot_iter=fewshot_iter,
        )


def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    label_column: str,
    label_choices: list[bool | int | str],
    num_variants: int,
    rng: Random,
    fewshot_iter: Iterator[list[dict]] | None = None,
) -> dict[str, Any]:
    """Prompt-generating function to pass to `IterableDataset.map`."""
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
    label = example[label_column]

    for template in templates:
        choices = []

        for pseudo_label in label_choices:
            fake_example = example.copy()
            fake_example[label_column] = pseudo_label

            q, a = template.apply(fake_example)
            prompt_counter[(q, a)] += 1

            if fewshot_iter is not None:
                # Infinite iterator so we don't need to worry about StopIteration
                fewshot_examples = next(fewshot_iter)
                fewshot_texts = [
                    qa_cat(q, a) for q, a in map(template.apply, fewshot_examples)
                ]
                q = "\n\n".join(fewshot_texts) + "\n\n" + q

            choices.append(
                dict(
                    # Strip whitespace from the answer to make it easier to
                    # compare with the model's output
                    answer=a.strip(),
                    question=q,
                )
            )

        prompts.append(choices)

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    # Our reporter training and evaluation code assumes that the labels are integers.
    # If they're not, we need to convert them with index(). label_choices is guaranteed
    # to be sorted (see above).
    return dict(
        label=label_choices.index(label),
        prompts=prompts,
        template_names=[template.name for template in templates],
    )
