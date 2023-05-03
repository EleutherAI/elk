"""Functions for extracting the hidden states of a model."""
import logging
import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import InitVar, dataclass, replace
from itertools import zip_longest
from typing import Any, Iterable, Literal
from warnings import filterwarnings

import torch
from datasets import (
    Array2D,
    Array3D,
    DatasetDict,
    DatasetInfo,
    DownloadMode,
    Features,
    Sequence,
    SplitDict,
    SplitInfo,
    Value,
    get_dataset_config_info,
)
from simple_parsing import Serializable, field
from torch import Tensor
from transformers import AutoConfig, PreTrainedModel

from ..promptsource import DatasetTemplates
from ..utils import (
    Color,
    assert_type,
    colorize,
    float_to_int16,
    infer_label_column,
    infer_num_classes,
    instantiate_model,
    instantiate_tokenizer,
    is_autoregressive,
    prevent_name_conflicts,
    select_split,
    select_train_val_splits,
    select_usable_devices,
)
from .dataset_name import (
    DatasetDictWithName,
    parse_dataset_string,
)
from .generator import _GeneratorBuilder
from .prompt_loading import load_prompts


@dataclass
class Extract(Serializable):
    """Config for extracting hidden states from a language model."""

    model: str = field(positional=True)
    """HF model string identifying the language model to extract hidden states from."""

    datasets: tuple[str, ...] = field(positional=True)
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"`"""

    data_dirs: tuple[str, ...] = ()
    """Directory to use for caching the hiddens. Defaults to `HF_DATASETS_CACHE`."""

    binarize: bool = False
    """Whether to binarize the dataset labels for multi-class datasets."""

    int8: bool = False
    """Whether to perform inference in mixed int8 precision with `bitsandbytes`."""

    max_examples: tuple[int, int] = (1000, 1000)
    """Maximum number of examples to use from each split of the dataset."""

    num_shots: int = 0
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    num_variants: int = -1
    """The number of prompt templates to use for each example. If -1, all available
    templates are used."""

    layers: tuple[int, ...] = ()
    """Indices of layers to extract hidden states from. We follow the HF convention, so
    0 is the embedding, and 1 is the output of the first transformer layer."""

    layer_stride: InitVar[int] = 1
    """Shortcut for `layers = (0,) + tuple(range(1, num_layers + 1, stride))`."""

    seed: int = 42
    """Seed to use for prompt randomization. Defaults to 42."""

    template_path: str | None = None
    """Path to pass into `DatasetTemplates`. By default we use the dataset name."""

    token_loc: Literal["first", "last", "mean"] = "last"
    """The location of the token to extract hidden states from."""

    use_encoder_states: bool = False
    """Whether to extract hidden states from the encoder instead of the decoder in the
    case of encoder-decoder models."""

    def __post_init__(self, layer_stride: int):
        if len(self.datasets) == 0:
            raise ValueError(
                "Must specify at least one dataset to extract hiddens from."
            )

        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )
        if not self.max_examples:
            self.max_examples = (int(1e100), int(1e100))

        # Broadcast the dataset name to all data_dirs
        if len(self.data_dirs) == 1:
            self.data_dirs *= len(self.datasets)
        elif self.data_dirs and len(self.data_dirs) != len(self.datasets):
            raise ValueError(
                "data_dirs should be a list of length 0, 1, or len(datasets),"
                f" but got {len(self.data_dirs)}"
            )

        if self.layers and layer_stride > 1:
            raise ValueError(
                "Cannot use both --layers and --layer-stride. Please use only one."
            )
        elif layer_stride > 1:
            from transformers import AutoConfig, PretrainedConfig

            # Look up the model config to get the number of layers
            config = assert_type(
                PretrainedConfig, AutoConfig.from_pretrained(self.model)
            )
            # Note that we always include 0 which is the embedding layer
            layer_range = range(1, config.num_hidden_layers + 1, layer_stride)
            self.layers = (0,) + tuple(layer_range)

    def explode(self) -> list["Extract"]:
        """Explode this config into a list of configs, one for each layer."""
        return [
            replace(self, datasets=(ds,), data_dirs=(data_dir,) if data_dir else ())
            for ds, data_dir in zip_longest(self.datasets, self.data_dirs)
        ]


@torch.inference_mode()
def extract_hiddens(
    cfg: "Extract",
    *,
    device: str | torch.device = "cpu",
    split_type: Literal["train", "val"] = "train",
    rank: int = 0,
    world_size: int = 1,
) -> Iterable[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Silence datasets logging messages from all but the first process
    if rank != 0:
        filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    ds_names = cfg.datasets
    assert len(ds_names) == 1, "Can only extract hiddens from one dataset at a time."

    # We use contextlib.redirect_stdout to prevent `bitsandbytes` from printing its
    # welcome message on every rank
    with redirect_stdout(None) if rank != 0 else nullcontext():
        model = instantiate_model(cfg.model, device=device, load_in_8bit=cfg.int8)
        tokenizer = instantiate_tokenizer(
            cfg.model, truncation_side="left", verbose=rank == 0
        )

    is_enc_dec = model.config.is_encoder_decoder
    if is_enc_dec and cfg.use_encoder_states:
        assert hasattr(model, "get_encoder") and callable(model.get_encoder)
        model = assert_type(PreTrainedModel, model.get_encoder())
        is_enc_dec = False

    has_lm_preds = is_autoregressive(model.config, not cfg.use_encoder_states)
    if has_lm_preds and rank == 0:
        print("Model has language model head, will store predictions.")

    prompt_ds = load_prompts(
        ds_names[0],
        binarize=cfg.binarize,
        num_shots=cfg.num_shots,
        num_variants=cfg.num_variants,
        split_type=split_type,
        template_path=cfg.template_path,
        rank=rank,
        world_size=world_size,
    )

    # Add one to the number of layers to account for the embedding layer
    layer_indices = cfg.layers or tuple(range(model.config.num_hidden_layers + 1))

    global_max_examples = cfg.max_examples[0 if split_type == "train" else 1]

    # break `max_examples` among the processes roughly equally
    max_examples = global_max_examples // world_size
    max_length = assert_type(int, tokenizer.model_max_length)

    # Keep track of the number of examples we've yielded so far. We can't do something
    # clean like `islice` the dataset, because we skip examples that are too long, and
    # we can't predict how many of those there will be.
    num_yielded = 0

    # the last process gets the remainder (which is usually small)
    if rank == world_size - 1:
        max_examples += global_max_examples % world_size

    for example in prompt_ds:
        # Check if we've yielded enough examples
        if num_yielded >= max_examples:
            break

        num_variants = len(example["prompts"])
        num_choices = len(example["prompts"][0])

        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                num_variants,
                num_choices,
                model.config.hidden_size,
                device=device,
                dtype=torch.int16,
            )
            for layer_idx in layer_indices
        }
        lm_logits = torch.empty(
            num_variants,
            num_choices,
            device=device,
            dtype=torch.float32,
        )
        text_questions = []

        # Iterate over variants
        for i, record in enumerate(example["prompts"]):
            variant_questions = []

            # Iterate over answers
            for j, choice in enumerate(record):
                text = choice["question"]

                # Only feed question, not the answer, to the encoder for enc-dec models
                target = choice["answer"] if is_enc_dec else None
                encoding = tokenizer(
                    text,
                    # Keep [CLS] and [SEP] for BERT-style models
                    add_special_tokens=True,
                    return_tensors="pt",
                    text_target=target,  # type: ignore[arg-type]
                ).to(device)

                ids = assert_type(Tensor, encoding.input_ids)
                if is_enc_dec:
                    answer = labels = assert_type(Tensor, encoding.labels)
                else:
                    encoding2 = tokenizer(
                        choice["answer"],
                        # Don't include [CLS] and [SEP] in the answer
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).to(device)

                    answer = assert_type(Tensor, encoding2.input_ids)
                    labels = (
                        # -100 is the mask token
                        torch.cat([torch.full_like(ids, -100), answer], dim=-1)
                        if has_lm_preds
                        else None
                    )
                    ids = torch.cat([ids, answer], -1)

                # If this input is too long, skip it
                if ids.shape[-1] > max_length:
                    break
                else:
                    # Record the EXACT question we fed to the model
                    variant_questions.append(text)

                inputs = dict(input_ids=ids.long(), labels=labels)
                outputs = model(**inputs, output_hidden_states=True)

                # Compute the log probability of the answer tokens if available
                if has_lm_preds:
                    lm_logits[i, j] = -assert_type(Tensor, outputs.loss)

                hiddens = (
                    outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                )
                # Throw out layers we don't care about
                hiddens = [hiddens[i] for i in layer_indices]

                # Current shape of each element: (batch_size, seq_len, hidden_size)
                if cfg.token_loc == "first":
                    hiddens = [h[..., 0, :] for h in hiddens]
                elif cfg.token_loc == "last":
                    hiddens = [h[..., -1, :] for h in hiddens]
                elif cfg.token_loc == "mean":
                    hiddens = [h.mean(dim=-2) for h in hiddens]
                else:
                    raise ValueError(f"Invalid token_loc: {cfg.token_loc}")

                for layer_idx, hidden in zip(layer_indices, hiddens):
                    hidden_dict[f"hidden_{layer_idx}"][i, j] = float_to_int16(hidden)

            # We skipped a pseudolabel because it was too long; break out of this whole
            # example and move on to the next one
            if len(variant_questions) != num_choices:
                break

            # Usual case: we have the expected number of pseudolabels
            text_questions.append(variant_questions)

        # We skipped a variant because it was too long; move on to the next example
        if len(text_questions) != num_variants:
            continue

        out_record: dict[str, Any] = dict(
            label=example["label"],
            variant_ids=example["template_names"],
            text_questions=text_questions,
            **hidden_dict,
        )
        if has_lm_preds:
            out_record["model_logits"] = lm_logits

        num_yielded += 1
        yield out_record


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def hidden_features(cfg: Extract) -> tuple[DatasetInfo, Features]:
    """Return the HuggingFace `Features` corresponding to an `Extract` config."""
    with prevent_name_conflicts():
        model_cfg = AutoConfig.from_pretrained(cfg.model)

    ds_name, config_name = parse_dataset_string(dataset_config_str=cfg.datasets[0])
    info = get_dataset_config_info(ds_name, config_name or None)

    if not cfg.template_path:
        prompter = DatasetTemplates(ds_name, config_name)
    else:
        prompter = DatasetTemplates(cfg.template_path)

    ds_features = assert_type(Features, info.features)
    label_col = prompter.label_column or infer_label_column(ds_features)
    num_classes = (
        2
        if cfg.binarize or prompter.binarize
        else infer_num_classes(ds_features[label_col])
    )

    num_variants = cfg.num_variants
    if num_variants < 0:
        num_dropped = prompter.drop_non_mc_templates()
        num_variants = len(prompter.templates)
        if num_dropped:
            print(f"Dropping {num_dropped} non-multiple choice templates")

    layer_cols = {
        f"hidden_{layer}": Array3D(
            dtype="int16",
            shape=(num_variants, num_classes, model_cfg.hidden_size),
        )
        # Add 1 to include the embedding layer
        for layer in cfg.layers or range(model_cfg.num_hidden_layers + 1)
    }
    other_cols = {
        "variant_ids": Sequence(
            Value(dtype="string"),
            length=num_variants,
        ),
        "label": Value(dtype="int64"),
        "text_questions": Sequence(
            Sequence(
                Value(dtype="string"),
            ),
            length=num_variants,
        ),
    }

    # Only add model_logits if the model is an autoregressive model
    if is_autoregressive(model_cfg, not cfg.use_encoder_states):
        other_cols["model_logits"] = Array2D(
            shape=(num_variants, num_classes),
            dtype="float32",
        )

    return info, Features({**layer_cols, **other_cols})


def extract(
    cfg: "Extract",
    *,
    disable_cache: bool = False,
    highlight_color: Color = "cyan",
    num_gpus: int = -1,
    min_gpu_mem: int | None = None,
    split_type: Literal["train", "val", None] = None,
) -> DatasetDictWithName:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""
    info, features = hidden_features(cfg)

    devices = select_usable_devices(num_gpus, min_memory=min_gpu_mem)
    limits = cfg.max_examples
    splits = assert_type(SplitDict, info.splits)

    pretty_name = colorize(assert_type(str, cfg.datasets[0]), highlight_color)
    if split_type is None:
        train, val = select_train_val_splits(splits)

        print(f"{pretty_name} using '{train}' for training and '{val}' for validation")
        splits = SplitDict({train: splits[train], val: splits[val]})
        split_types = ["train", "val"]
    else:
        # Remove the split we're not using
        limits = [limits[0]] if split_type == "train" else limits
        split_name = select_split(splits, split_type)
        splits = SplitDict({split_name: splits[split_name]})
        split_types = [split_type]

        if split_type == "train":
            print(f"{pretty_name} using '{split_name}' for training")
        else:
            print(f"{pretty_name} using '{split_name}' for validation")

    builders = {
        split_name: _GeneratorBuilder(
            cache_dir=None,
            features=features,
            generator=_extraction_worker,
            split_name=split_name,
            split_info=SplitInfo(
                name=split_name,
                num_examples=min(limit, v.num_examples) * len(cfg.datasets),
                dataset_name=v.dataset_name,
            ),
            gen_kwargs=dict(
                cfg=[cfg] * len(devices),
                device=devices,
                rank=list(range(len(devices))),
                split_type=[ty] * len(devices),
                world_size=[len(devices)] * len(devices),
            ),
        )
        for limit, (split_name, v), ty in zip(limits, splits.items(), split_types)
    }
    import multiprocess as mp

    mp.set_start_method("spawn", force=True)  # type: ignore[attr-defined]

    ds = dict()
    for split, builder in builders.items():
        builder.download_and_prepare(
            download_mode=DownloadMode.FORCE_REDOWNLOAD if disable_cache else None,
            num_proc=len(devices),
        )
        ds[split] = builder.as_dataset(split=split)

    dataset_dict = DatasetDict(ds)
    return DatasetDictWithName(
        name=cfg.datasets[0],
        dataset=dataset_dict,
    )
