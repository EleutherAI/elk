from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    binarize,
    infer_label_column,
    infer_num_classes,
    select_train_val_splits,
)
from .balanced_sampler import FewShotSampler
from dataclasses import dataclass
from datasets import (
    interleave_datasets,
    load_dataset,
    ClassLabel,
    Features,
    IterableDataset,
    Sequence,
)
from datasets.distributed import split_dataset_by_node
from random import Random
from simple_parsing.helpers import field, Serializable
from typing import Any, Iterator, Literal, Optional


@dataclass
class PromptConfig(Serializable):
    """
    Args:
        dataset: Space-delimited name of the HuggingFace dataset to use, e.g.
            `"super_glue boolq"` or `"imdb"`.
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
        seed: The seed to use for prompt randomization. Defaults to 42.
        num_variants: The number of prompt templates to apply to each predicate upon
            call to __getitem__. Use -1 to apply all available templates. Defaults to 1.
    """

    datasets: list[str] = field(positional=True)
    balance: bool = False
    data_dir: Optional[str] = None
    label_column: Optional[str] = None
    max_examples: list[int] = field(default_factory=lambda: [750, 250])
    num_shots: int = 0
    num_variants: int = -1
    seed: int = 42

    def __post_init__(self):
        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )


def load_prompts(
    *dataset_strings: str,
    max_examples: int = 0,
    num_shots: int = 0,
    seed: int = 42,
    split_type: Literal["train", "val"] = "train",
    rank: int = 0,
    world_size: int = 1,
) -> IterableDataset:
    """Load a dataset full of prompts generated from the specified datasets.

    Args:
        dataset_strings: Space-delimited names of the HuggingFace datasets to use,
            e.g. `"super_glue boolq"` or `"imdb"`.
        max_examples: The maximum number of examples to use from the dataset.
        num_shots: The number of examples to use in few-shot prompts. If zero, prompts
            are zero-shot.
        seed: The seed to use for prompt randomization.
        split_type: Whether to use the train or val split of the dataset.
        rank: The rank of the current process. Defaults to 0.
        world_size: The number of processes. Defaults to 1.

    Returns:
        An iterable dataset of prompts.
    """
    prompt_datasets = []
    prompters = []
    raw_datasets = []
    train_datasets = []
    rng = Random(seed)

    # First load the datasets and prompters. We need to know the minimum number of
    # templates for any dataset in order to make sure we don't run out of prompts.
    for ds_string in dataset_strings:
        ds_name, _, config_name = ds_string.partition(" ")
        prompters.append(DatasetTemplates(ds_name, config_name))

        ds_dict = assert_type(
            dict, load_dataset(ds_name, config_name or None, streaming=True)
        )
        train_name, val_name = select_train_val_splits(ds_dict)
        split_name = val_name if split_type == "val" else train_name
        raw_datasets.append(assert_type(IterableDataset, ds_dict[split_name]))
        train_datasets.append(assert_type(IterableDataset, ds_dict[train_name]))

    num_variants = min(len(prompter.templates) for prompter in prompters)
    for ds, train_ds, prompter in zip(raw_datasets, train_datasets, prompters):
        label_column = infer_label_column(ds.features)
        num_classes = infer_num_classes(ds.features[label_column])

        # Remove everything but the label column
        extra_cols = list(assert_type(Features, ds.features))
        extra_cols.remove(label_column)

        if label_column != "label":
            ds = ds.rename_column(label_column, "label")
        if num_shots > 0:
            fewshot = FewShotSampler(
                train_ds,
                num_shots=num_shots,
                rng=rng,
            )
            fewshot_iter = iter(fewshot)
        else:
            fewshot_iter = None

        # Canonicalize the name and dtype of the label column
        ds = ds.map(
            _convert_to_prompts,
            fn_kwargs=dict(
                label_column=label_column,
                num_classes=num_classes,
                num_variants=num_variants,
                prompter=prompter,
                rng=rng,
                fewshot_iter=fewshot_iter,
            ),
            remove_columns=extra_cols,
        ).map(
            # Add the builder and config name to the records directly to make
            # sure we don't forget what dataset they came from.
            lambda _: dict(
                builder_name=ds.info.builder_name,
                config_name=ds.info.config_name,
            ),
            # Explicit typing makes interleave_datasets work a lot faster
            features=Features(
                {
                    label_column: ClassLabel(names=["neg", "pos"]),
                    "builder_name": "string",
                    "config_name": "string",
                    "prompts": Sequence(
                        Sequence(
                            {"answer": "string", "text": "string"},
                            length=num_classes,
                        ),
                        length=num_variants,
                    ),
                    "template_names": Sequence("string"),
                }
            ),
        )
        prompt_datasets.append(ds)

    master_ds = interleave_datasets(prompt_datasets)
    if max_examples > 0:
        master_ds = master_ds.take(max_examples)
    if world_size > 1:
        # This prints to stdout which is slightly annoying
        master_ds = split_dataset_by_node(master_ds, rank, world_size)

    return master_ds


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
    prompts = []
    templates = list(prompter.templates.values())
    if num_variants < len(templates):
        templates = rng.sample(templates, num_variants)

    def qa_cat(q: str, a: str) -> str:
        # if the jinja template already adds whitespace, don't add more
        sep = "" if not q or q[-1].isspace() or not a or a[0].isspace() else " "
        return f"{q}{sep}{a}" if a and not a.isspace() else q

    for template in templates:
        choices = []

        if num_classes > 2:
            template, _ = binarize(
                template, example[label_column], rng.choice([0, 1]), rng
            )

        for answer_idx in range(num_classes):
            fake_example = example.copy()
            fake_example[label_column] = answer_idx

            q, a = template.apply(fake_example)
            text = qa_cat(q, a)

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

    return dict(
        prompts=prompts,
        template_names=prompter.all_template_names,
    )
