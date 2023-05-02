from collections import Counter
from random import Random
from typing import Any, Iterator, Literal

from datasets import ClassLabel, Dataset, Value, load_dataset

from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    infer_label_column,
    select_split,
)
from .balanced_sampler import BalancedSampler, FewShotSampler


def load_prompts(
    ds_string: str,
    *,
    binarize: bool = False,
    num_shots: int = 0,
    num_variants: int = -1,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    template_path: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[dict]:
    """Load a dataset full of prompts generated from the specified dataset.

    Args:
        ds_string: Name of HF dataset to use, e.g. `"super_glue:boolq"` or `"imdb"`.
        binarize: Whether to binarize the dataset labels for multi-class datasets.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        template_path: Path to feed into `DatasetTemplates` for loading templates.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(":")

    ds_dict = assert_type(dict, load_dataset(ds_name, config_name or None))
    split_name = select_split(ds_dict, split_type)

    ds = assert_type(Dataset, ds_dict[split_name].shuffle(seed=seed))
    if world_size > 1:
        ds = ds.shard(world_size, rank)

    if template_path is None:
        prompter = DatasetTemplates(ds_name, config_name)
    else:
        prompter = DatasetTemplates(template_path)

    # If the prompt template says to binarize, we should
    binarize = binarize or prompter.binarize
    prompter.drop_non_mc_templates()

    num_templates = len(prompter.templates)
    num_variants = (
        num_templates if num_variants == -1 else min(num_variants, num_templates)
    )
    assert num_variants > 0
    if rank == 0:
        print(f"Using {num_variants} variants of each prompt")

    label_column = prompter.label_column or infer_label_column(ds.features)

    label_feature = ds.features[label_column]
    if isinstance(label_feature, ClassLabel):
        label_choices = [label_feature.str2int(label) for label in label_feature.names]
    elif isinstance(label_feature, Value) and label_feature.dtype == "bool":
        label_choices = [False, True]
    else:
        # Which classes are actually present in this split of the dataset?
        # This is shockingly fast since it uses an optimized Apache Arrow primitive.
        label_choices = sorted(ds.unique(label_column))
        if rank == 0:
            print(f"Using the following pseudo-labels: {label_choices}")

    rng = Random(seed)
    if num_shots > 0:
        train_name = select_split(ds_dict, "train")
        fewshot = FewShotSampler(
            ds_dict[train_name].shuffle(seed=seed),  # TODO: not iterator
            num_shots=num_shots,
            rng=rng,
        )
        fewshot_iter = iter(fewshot)
    else:
        fewshot_iter = None

    if label_column in ds.features:
        ds = BalancedSampler(
            ds.to_iterable_dataset(),
            set(label_choices),
            label_col=label_column,
        )
    else:
        if rank == 0:
            print("No label column found, not balancing")
        ds = ds.to_iterable_dataset()

    for example in ds:
        yield _convert_to_prompts(
            example,
            binarize=binarize,
            label_column=label_column,
            label_choices=label_choices,  # type: ignore[arg-type]
            num_variants=num_variants,
            prompter=prompter,
            rng=rng,
            fewshot_iter=fewshot_iter,
        )


def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    binarize: bool,
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

    if binarize:
        # Replace the full list of possibilities with a randomly sampled false label
        # and the correct label, as done in the DLK paper. Note that this does add some
        # "supervision" by stacking the deck in favor of the correct answer.
        label_choices = [
            rng.choice([c for c in label_choices if c != label]),
            label,
        ]
        rng.shuffle(label_choices)

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
