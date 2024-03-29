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
    num_shots: int = 0,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    template_path: str | None = None,
    include_answers: bool = False,
    balance: bool = True,
    statement_column: str | None = None,
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
        statement_column: Name of the column to use for the statement text.

    Returns:
        An iterable of prompt dictionaries.
    """
    ds_name, _, config_name = ds_string.partition(":")

    ds_dict = assert_type(dict, load_dataset(ds_name, config_name or None))
    split_name = select_split(ds_dict, split_type)

    ds = assert_type(Dataset, ds_dict[split_name])
    if "row_id" not in ds.column_names:
        ds = ds.add_column("row_id", range(len(ds)))  # type: ignore
    else:
        print("Found `row_id` column, using it as the example id")
    ds = ds.shuffle(seed=seed)

    prompter, using_blank = get_prompter(ds_name, config_name, template_path)
    if using_blank:
        print('Using blank template "{{ statement }}".')
        statement_column = statement_column or "statement"
        if statement_column not in ds.column_names:
            raise ValueError(
                f'Could not find statement column "{statement_column}".'
                f" Please include the column or specify a different one with the"
                f" `statement_column` argument."
            )
        if statement_column != "statement":
            ds = ds.rename_column(statement_column, "statement")

    # TODO: allow for optionally using contrast pair templates so people
    # don't have to rewrite them

    num_templates = len(prompter.templates)
    assert num_templates > 0

    print(f"Extracting {num_templates} variants of each prompt")

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
        print(f"Using the following pseudo-labels: {label_choices}")

    rng = Random(seed)
    if num_shots > 0:
        train_name = select_split(ds_dict, "train")
        fewshot = FewShotSampler(
            ds_dict[train_name].shuffle(seed=seed),
            num_shots=num_shots,
            rng=rng,
        )
        fewshot_iter = iter(fewshot)
    else:
        fewshot_iter = None

    if label_column in ds.features and balance:
        print(f"Balancing dataset by {label_column}")
        ds = BalancedSampler(
            ds.to_iterable_dataset(),
            set(label_choices),
            label_col=label_column,
        )
    else:
        if balance:
            print("No label column found, not balancing")
        ds = ds.to_iterable_dataset()

    for example in ds:
        yield _convert_to_prompts(
            example,
            label_column=label_column,
            label_choices=label_choices,  # type: ignore[arg-type]
            prompter=prompter,
            include_answers=include_answers,
            fewshot_iter=fewshot_iter,
        )


def _convert_to_prompts(
    example: dict[str, Any],
    prompter: DatasetTemplates,
    label_column: str,
    label_choices: list[bool | int | str],
    include_answers: bool = False,
    fewshot_iter: Iterator[list[dict]] | None = None,
) -> dict[str, Any]:
    """Prompt-generating function to pass to `IterableDataset.map`."""
    statements = []
    templates = list(prompter.templates.values())

    # For sanity checking that prompts are unique
    prompt_counter = Counter()
    label = example[label_column]

    for template in templates:
        statement = template.apply(example)

        choices = template.get_fixed_answer_choices_list()
        choices = tuple(choices) if choices is not None else None
        prompt_counter[(statement, choices)] += 1

        if fewshot_iter is not None:
            # Infinite iterator so we don't need to worry about StopIteration
            fewshot_examples = next(fewshot_iter)
            fewshot_texts = list(map(template.apply, fewshot_examples))
            statement = "\n\n".join(fewshot_texts) + "\n\n" + statement

        statements.append(statement)

    # Sanity check: variants should be unique
    ((maybe_dup, dup_count),) = prompt_counter.most_common(1)
    if dup_count > 1:
        raise ValueError(f'Prompt duplicated {dup_count} times! "{maybe_dup}"')

    # Our reporter training and evaluation code assumes that the labels are integers.
    # If they're not, we need to convert them with index(). label_choices is guaranteed
    # to be sorted (see above).
    out_dict = dict(
        row_id=example["row_id"],
        label=label_choices.index(label),
        statements=statements,
        template_names=[template.name for template in templates],
    )
    if include_answers:
        out_dict.update(
            answer_choices=[
                template.get_fixed_answer_choices_list() for template in templates
            ],
            suffixes=[template.suffix for template in templates],
        )
    return out_dict


def get_prompter(
    ds_name: str, config_name: str | None, template_path: str | None = None
) -> tuple[DatasetTemplates, bool]:
    if template_path is None:
        try:
            return DatasetTemplates(ds_name, config_name), False
        except ValueError:
            return DatasetTemplates("_default"), True
    return DatasetTemplates(template_path), template_path == "_default"
