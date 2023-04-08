"""Functions for extracting the hidden states of a model."""
import logging
import os
from dataclasses import InitVar, dataclass
from itertools import islice
from typing import Iterable, Literal, Optional, Union

import torch
from datasets import (
    Array3D,
    ClassLabel,
    DatasetDict,
    Features,
    Sequence,
    SplitDict,
    SplitInfo,
    Value,
    get_dataset_config_info,
)
from simple_parsing import Serializable, field
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from elk.rnn.elmo import ElmoConfig, ElmoModel, ElmoTokenizer
from elk.utils.typing import float32_to_int16

from ..utils import (
    assert_type,
    select_train_val_splits,
    select_usable_devices,
)
from .balanced_sampler import BalancedSampler
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
        min_gpu_mem: Minimum amount of free memory (in bytes) required to select a GPU.
    """

    prompts: PromptConfig
    model: str = field(positional=True)

    layers: tuple[int, ...] = ()
    layer_stride: InitVar[int] = 1
    token_loc: Literal["first", "last", "mean"] = "last"
    min_gpu_mem: Optional[int] = None
    num_gpus: int = -1

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

    def execute(self):
        extract(cfg=self, num_gpus=self.num_gpus)


@torch.no_grad()
def extract_hiddens(
    cfg: "Extract",
    *,
    device: Union[str, torch.device] = "cpu",
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

    prompt_ds = load_prompts(
        *cfg.prompts.datasets,
        split_type=split_type,
        stream=cfg.prompts.stream,
        rank=rank,
        world_size=world_size,
    )  # this dataset is already sharded, but hasn't been truncated to max_examples

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    model = None
    if cfg.model == "elmo":
        model = ElmoModel.from_pretrained("").to(device)
    else:
        model = AutoModel.from_pretrained(
            cfg.model, torch_dtype="auto" if device != "cpu" else torch.float32
        ).to(device)

    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    tokenizer = None
    if cfg.model == "elmo":
        tokenizer = ElmoTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, truncation_side="left", verbose=False
        )
    is_enc_dec = model.config.is_encoder_decoder

    # If this is an encoder-decoder model we don't need to run the decoder at all.
    # Just strip it off, making the problem equivalent to a regular encoder-only model.
    if is_enc_dec:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = assert_type(PreTrainedModel, model.get_encoder())

    # Iterating over questions
    layer_indices = cfg.layers or tuple(range(model.config.num_hidden_layers))
    # print(f"Using {prompt_ds} variants for each dataset")

    global_max_examples = cfg.prompts.max_examples[0 if split_type == "train" else 1]
    # break `max_examples` among the processes roughly equally
    max_examples = global_max_examples // world_size
    # the last process gets the remainder (which is usually small)
    if rank == world_size - 1:
        max_examples += global_max_examples % world_size

    print(f"Extracting {max_examples} examples from {prompt_ds} on {device}")

    for example in islice(BalancedSampler(prompt_ds), max_examples):
        num_variants = len(example["prompts"])
        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                num_variants,
                2,  # contrast pair
                model.config.hidden_size,
                device=device,
                dtype=torch.int16,
            )
            for layer_idx in layer_indices
        }
        text_inputs = []

        # Iterate over variants
        for i, record in enumerate(example["prompts"]):
            variant_inputs = []

            # Iterate over answers
            for j in range(2):
                text = record[j]["text"]
                variant_inputs.append(text)

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                ).to(device)
                outputs = None
                if cfg.model == "elmo":
                    outputs = model(inputs)
                else:
                    model(**inputs, output_hidden_states=True)

                hiddens = (
                    outputs
                    if cfg.model == "elmo"
                    else (
                        outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                    )
                )
                # First element of list is the input embeddings
                hiddens = outputs if cfg.model == "elmo" else hiddens[1:]

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
                    hidden_dict[f"hidden_{layer_idx}"][i, j] = float32_to_int16(hidden)

            text_inputs.append(variant_inputs)

        yield dict(
            label=example["label"],
            variant_ids=example["template_names"],
            text_inputs=text_inputs,
            **hidden_dict,
        )


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def extract(cfg: "Extract", num_gpus: int = -1) -> DatasetDict:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""

    def get_splits() -> SplitDict:
        available_splits = assert_type(SplitDict, info.splits)
        train_name, val_name = select_train_val_splits(available_splits)
        print(f"Using '{train_name}' for training and '{val_name}' for validation")

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

    model_cfg = (
        ElmoConfig() if cfg.model == "elmo" else AutoConfig.from_pretrained(cfg.model)
    )
    num_variants = cfg.prompts.num_variants

    ds_name, _, config_name = cfg.prompts.datasets[0].partition(" ")
    info = get_dataset_config_info(ds_name, config_name or None)

    layer_cols = {
        f"hidden_{layer}": Array3D(
            dtype="int16",
            shape=(num_variants, 2, model_cfg.hidden_size),
        )
        for layer in cfg.layers or range(model_cfg.num_hidden_layers)
    }
    other_cols = {
        "variant_ids": Sequence(
            Value(dtype="string"),
            length=num_variants,
        ),
        "label": ClassLabel(names=["neg", "pos"]),
        "text_inputs": Sequence(
            Sequence(
                Value(dtype="string"),
                length=2,
            ),
            length=num_variants,
        ),
    }
    devices = select_usable_devices(num_gpus, min_memory=cfg.min_gpu_mem)
    builders = {
        split_name: _GeneratorBuilder(
            cache_dir=None,
            features=Features({**layer_cols, **other_cols}),
            generator=_extraction_worker,
            split_name=split_name,
            split_info=split_info,
            gen_kwargs=dict(
                cfg=[cfg] * len(devices),
                device=devices,
                rank=list(range(len(devices))),
                split_type=[split_name] * len(devices),
                world_size=[len(devices)] * len(devices),
            ),
        )
        for (split_name, split_info) in get_splits().items()
    }

    ds = dict()
    for split, builder in builders.items():
        builder.download_and_prepare(num_proc=len(devices))
        ds[split] = builder.as_dataset(split=split)

    return DatasetDict(ds)
