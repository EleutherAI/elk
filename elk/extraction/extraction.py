"""Functions for extracting the hidden states of a model."""
import json
import logging
import os
from copy import copy
from dataclasses import InitVar, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from datasets import (
    Array2D,
    Array3D,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    SplitDict,
    SplitInfo,
    Value,
    get_dataset_config_info,
    load_from_disk,
)
from simple_parsing import Serializable, field
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from ..promptsource import DatasetTemplates
from ..utils import (
    assert_type,
    convert_span,
    float32_to_int16,
    infer_label_column,
    infer_num_classes,
    instantiate_model,
    is_autoregressive,
    select_train_val_splits,
    select_usable_devices,
)
from .generator import _GeneratorBuilder
from .prompt_loading import PromptConfig, load_prompts


@dataclass
class Extract(Serializable):
    """
    Args:
        model: HuggingFace model string identifying the language model to extract
            hidden states from.
        prompts: The configuration for the prompt prompts.
        layers: The layers to extract hidden states from.
        layer_stride: Shortcut for setting `layers` to `range(0, num_layers, stride)`.
        token_loc: The location of the token to extract hidden states from. Can be
            either "first", "last", or "mean". Defaults to "last".
    """

    prompts: PromptConfig
    model: str = field(positional=True)

    layers: tuple[int, ...] = ()
    layer_stride: InitVar[int] = 1
    token_loc: Literal["first", "last", "mean"] = "last"

    def __post_init__(self, layer_stride: int):
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
            self.layers = tuple(range(0, config.num_hidden_layers, layer_stride))

    def explode(self) -> list["Extract"]:
        """Explode this config into a list of configs, one for each layer."""
        copies = []

        for prompt_cfg in self.prompts.explode():
            cfg = copy(self)
            cfg.prompts = prompt_cfg
            copies.append(cfg)

        return copies


@torch.no_grad()
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
        logging.disable(logging.CRITICAL)

    p_cfg = cfg.prompts
    ds_names = p_cfg.datasets
    assert len(ds_names) == 1, "Can only extract hiddens from one dataset at a time."

    prompt_ds = load_prompts(
        ds_names[0],
        label_column=p_cfg.label_columns[0] if p_cfg.label_columns else None,
        num_classes=p_cfg.num_classes,
        split_type=split_type,
        stream=p_cfg.stream,
        rank=rank,
        world_size=world_size,
    )  # this dataset is already sharded, but hasn't been truncated to max_examples

    model = instantiate_model(
        cfg.model, torch_dtype="auto" if device != "cpu" else torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, truncation_side="left", verbose=False
    )
    has_lm_preds = is_autoregressive(model.config)
    if has_lm_preds and rank == 0:
        print("Model has language model head, will store predictions.")

    # Iterating over questions
    layers = cfg.layers or tuple(range(model.config.num_hidden_layers))

    global_max_examples = p_cfg.max_examples[0 if split_type == "train" else 1]
    max_examples = get_max_examples(global_max_examples, rank, world_size)

    for example in islice(prompt_ds, max_examples):
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
            for layer_idx in layers
        }
        lm_preds = torch.empty(
            num_variants,
            num_choices,
            device=device,
            dtype=torch.float32,
        )
        text_inputs = []

        # Iterate over variants
        for i, record in enumerate(example["prompts"]):
            variant_inputs = []

            # Iterate over answers
            for j, choice in enumerate(record):
                text = choice["text"]

                # TODO: Do something smarter than "rindex" here. Really we want to
                # get the span of the answer directly from Jinja, but that doesn't
                # seem possible. This approach may fail for complex templates.
                answer_start = text.rindex(choice["answer"])

                # Only feed question, not the answer, to the encoder for enc-dec models
                if model.config.is_encoder_decoder:
                    # TODO: Maybe make this more generic for complex templates?
                    text = text[:answer_start].rstrip()
                    target = choice["answer"]
                else:
                    target = None

                # Record the EXACT string we fed to the model
                variant_inputs.append(text)
                inputs = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    text_target=target,  # type: ignore[arg-type]
                    truncation=True,
                )

                # The offset_mapping is a sorted list of (start, end) tuples. We locate
                # the start of the answer in the tokenized sequence with binary search.
                offsets = inputs.pop("offset_mapping").squeeze().tolist()
                inputs = inputs.to(device)

                # Run the forward pass
                outputs = model(**inputs, output_hidden_states=True)

                # Compute the log probability of the answer tokens if available
                if has_lm_preds:
                    start, end = convert_span(
                        offsets, (answer_start, answer_start + len(choice["answer"]))
                    )
                    log_p = outputs.logits[..., start - 1 : end - 1, :].log_softmax(
                        dim=-1
                    )
                    tokens = inputs.input_ids[..., start:end, None]
                    lm_preds[i, j] = log_p.gather(-1, tokens).sum()

                elif isinstance(outputs, Seq2SeqLMOutput):
                    # The cross entropy loss is averaged over tokens, so we need to
                    # multiply by the length to get the total log probability.
                    length = inputs.labels.shape[-1]
                    lm_preds[i, j] = -assert_type(Tensor, outputs.loss) * length

                hiddens = (
                    outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                )

                hiddens = select_hiddens(hiddens, cfg.token_loc, layers)

                for layer, hidden in zip(layers, hiddens):
                    hidden_dict[f"hidden_{layer}"][i, j] = float32_to_int16(hidden)

            text_inputs.append(variant_inputs)

        out_record: dict[str, Any] = dict(
            label=example["label"],
            variant_ids=example["template_names"],
            text_inputs=text_inputs,
            **hidden_dict,
        )
        if has_lm_preds:
            out_record["model_preds"] = lm_preds.softmax(dim=-1)

        yield out_record


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def select_hiddens(
    hiddens: list[torch.Tensor], token_loc: str, layer_indices: tuple[int, ...]
) -> list[torch.Tensor]:
    # First element of list is the input embeddings
    hiddens = hiddens[1:]

    # Throw out layers we don't care about
    hiddens = [hiddens[i] for i in layer_indices]

    # Current shape of each element: (batch_size, seq_len, hidden_size)
    if token_loc == "first":
        hiddens = [h[..., 0, :] for h in hiddens]
    elif token_loc == "last":
        hiddens = [h[..., -1, :] for h in hiddens]
    elif token_loc == "mean":
        hiddens = [h.mean(dim=-2) for h in hiddens]
    else:
        raise ValueError(f"Invalid token_loc: {token_loc}")

    return hiddens


def get_max_examples(global_max_examples: int, rank: int, world_size: int) -> int:
    """Get the maximum number of examples to extract for a given process."""
    # break `max_examples` among the processes roughly equally
    max_examples = global_max_examples // world_size
    # the last process gets the remainder (which is usually small)
    if rank == world_size - 1:
        max_examples += global_max_examples % world_size

    return max_examples


def raw_extract_hiddens(
    cfg: "Extract",
    *,
    device: str | torch.device = "cpu",
    split_type: Literal["train", "val"] = "train",
    rank: int = 0,
    world_size: int = 1,
) -> Iterable[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states.

    This is a lightweight, functional version of the `Extractor` API.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Silence datasets logging messages from all but the first process
    if rank != 0:
        logging.disable(logging.CRITICAL)

    model = instantiate_model(
        cfg.model, torch_dtype="auto" if device != "cpu" else torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, truncation_side="left", verbose=False
    )
    has_lm_preds = is_autoregressive(model.config)
    if has_lm_preds and rank == 0:
        print("Model has language model head, will store predictions.")

    # Load the dataset
    ds = load_from_disk(assert_type(str, cfg.prompts.data_dirs[0]))
    ds = (
        assert_type(Dataset, ds)
        .shard(num_shards=world_size, index=rank)
        .shuffle(seed=cfg.prompts.seed)
    )

    # Iterating over questions
    layers = cfg.layers or tuple(range(model.config.num_hidden_layers))

    global_max_examples = cfg.prompts.max_examples[0 if split_type == "train" else 1]
    max_examples = get_max_examples(global_max_examples, rank, world_size)

    # TODO: fix balancing by merging changes from check-streamable
    for example in islice(ds, max_examples):
        text, label = example["text"], example["label"]  # type: ignore

        # prepend start token so that the model can predict the first token
        start_text = tokenizer.bos_token  # TODO: will this work for all models?
        text = start_text + text

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        # Run the forward pass
        outputs = model(**inputs, output_hidden_states=True)

        # Compute the log probability of the whole sequence
        if has_lm_preds:
            # compute the total log probability of the sequence
            # by summing the log probabilities of each token except the last
            # (which is predicting a new token)
            logprobs = outputs.logits[:, :-1, :].log_softmax(dim=-1)
            logprobs = logprobs.gather(
                dim=-1, index=inputs.input_ids[:, 1:].unsqueeze(-1)
            )
            total_logprob = logprobs.sum()
        else:
            total_logprob = None

        # Extract the hidden states
        hiddens = outputs.get("decoder_hidden_states") or outputs["hidden_states"]

        hiddens = select_hiddens(hiddens, cfg.token_loc, layers)

        record = {
            f"hidden_{layer}": float32_to_int16(hid.squeeze())
            for layer, hid in zip(layers, hiddens)
        }
        record["text_input"] = text
        record["label"] = label
        if total_logprob is not None:
            record["total_logprob"] = total_logprob.float().item()

        yield record


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _raw_extraction_worker(**kwargs):
    yield from raw_extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def extract(
    cfg: "Extract", num_gpus: int = -1, min_gpu_mem: int | None = None
) -> DatasetDict:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""

    def get_splits() -> SplitDict:
        available_splits = assert_type(SplitDict, info.splits)
        train_name, val_name = select_train_val_splits(available_splits)
        print(
            # Cyan color for dataset name
            f"\033[36m{info.builder_name}\033[0m: using '{train_name}' for training and"
            f" '{val_name}' for validation"
        )
        limit_list = cfg.prompts.max_examples

        return SplitDict(
            {
                k: SplitInfo(
                    name=k,
                    num_examples=min(limit, v.num_examples) * len(cfg.prompts.datasets),
                    dataset_name=v.dataset_name,
                )
                for limit, (k, v) in zip(limit_list, available_splits.items())
            },
            dataset_name=available_splits.dataset_name,
        )

    def get_raw_fingerprint() -> str:
        assert cfg.prompts.datasets == ["raw"]
        # grab the "_fingerprint" stored in "state.json"
        # of cfg.prompts.data_dirs[0]
        state_path = Path(cfg.prompts.data_dirs[0]) / "state.json"
        with open(state_path, "r") as f:
            return json.load(f)["_fingerprint"]

    model_cfg = AutoConfig.from_pretrained(cfg.model)

    is_raw = cfg.prompts.datasets == ["raw"]

    if is_raw:
        builder_name, config_name = "raw", "raw"

        split_dict = SplitDict(
            {
                "val": SplitInfo(
                    name="val",
                    num_examples=-1,  # TODO: Is this important?
                    dataset_name="raw",
                ),
            },
            dataset_name="raw",
        )

        layer_cols = {
            f"hidden_{layer}": Sequence(
                feature=Value(dtype="int16"),
                length=model_cfg.hidden_size,
            )
            for layer in cfg.layers or range(model_cfg.num_hidden_layers)
        }
        other_cols = {
            "label": ClassLabel(names=["False", "True"]),
            "text_input": Value(dtype="string"),
        }

        # Only add model_preds if the model is an autoregressive model
        if is_autoregressive(model_cfg):
            other_cols["total_logprob"] = Value(dtype="float32")
    else:
        # gather dataset info
        ds_name, _, config_name = cfg.prompts.datasets[0].partition(" ")
        info = get_dataset_config_info(ds_name, config_name or None)
        split_dict = get_splits()
        ds_features = assert_type(Features, info.features)
        label_col = (
            cfg.prompts.label_columns[0]
            if cfg.prompts.label_columns
            else infer_label_column(ds_features)
        )
        num_classes = cfg.prompts.num_classes or infer_num_classes(
            ds_features[label_col]
        )
        num_variants = cfg.prompts.num_variants
        if num_variants < 0:
            prompter = DatasetTemplates(ds_name, config_name)
            num_variants = len(prompter.templates)
        builder_name, config_name = info.builder_name, info.config_name

        # define the columns of the dataset
        layer_cols = {
            f"hidden_{layer}": Array3D(
                dtype="int16",
                shape=(num_variants, num_classes, model_cfg.hidden_size),
            )
            for layer in cfg.layers or range(model_cfg.num_hidden_layers)
        }
        other_cols = {
            "variant_ids": Sequence(
                Value(dtype="string"),
                length=num_variants,
            ),
            "label": Value(dtype="int64"),
            "text_inputs": Sequence(
                Sequence(
                    Value(dtype="string"),
                    length=2,
                ),
                length=num_variants,
            ),
        }

        # Only add model_preds if the model is an autoregressive model
        if is_autoregressive(model_cfg):
            other_cols["model_preds"] = Array2D(
                shape=(num_variants, num_classes),
                dtype="float32",
            )

    devices = select_usable_devices(num_gpus, min_memory=min_gpu_mem)
    builders = {
        split_name: _GeneratorBuilder(
            builder_name=builder_name,
            config_name=config_name,
            cache_dir=None,
            features=Features({**layer_cols, **other_cols}),
            generator=_raw_extraction_worker if is_raw else _extraction_worker,
            split_name=split_name,
            split_info=split_info,
            gen_kwargs=dict(
                cfg=[cfg] * len(devices),
                device=devices,
                rank=list(range(len(devices))),
                split_type=[split_name] * len(devices),
                world_size=[len(devices)] * len(devices),
            ),
            name=get_raw_fingerprint() if is_raw else None,  # for caching
        )
        for (split_name, split_info) in split_dict.items()
    }
    import multiprocess as mp

    mp.set_start_method("spawn", force=True)  # type: ignore[attr-defined]

    ds = dict()
    for split, builder in builders.items():
        builder.download_and_prepare(num_proc=len(devices))
        ds[split] = builder.as_dataset(split=split)

    return DatasetDict(ds)
