"""Functions for extracting the hidden states of a model."""

from .prompt_dataset import PromptDataset, PromptConfig
from ..utils import (
    assert_type,
    get_model_class,
    infer_label_column,
    select_usable_devices,
    float32_to_int16,
)
from .generator import _GeneratorBuilder
from bisect import bisect_left
from dataclasses import dataclass, InitVar
from datasets import (
    Array2D,
    Array3D,
    DatasetDict,
    Features,
    get_dataset_config_info,
    Sequence,
    Split,
    SplitDict,
    SplitInfo,
    Value,
)
from operator import itemgetter
from simple_parsing.helpers import field, Serializable
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
)
from typing import Iterable, Literal, Union
import logging
import torch


@dataclass
class ExtractionConfig(Serializable):
    """
    Args:
        model: HuggingFace model string identifying the language model to extract
            hidden states from.
        prompts: The configuration for the prompt prompts.
        layers: The layers to extract hidden states from.
        layer_stride: Shortcut for setting `layers` to `range(0, num_layers, stride)`.
        token_loc: The location of the token to extract hidden states from. Can be
            either "first", "last", or "mean". Defaults to "last".
        use_encoder_states: Whether to extract hiddens from the encoder in
            encoder-decoder models. Defaults to False.
    """

    prompts: PromptConfig
    model: str = field(positional=True)

    layers: tuple[int, ...] = ()
    layer_stride: InitVar[int] = 1
    token_loc: Literal["first", "last", "mean"] = "last"
    use_encoder_states: bool = False

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


@torch.no_grad()
def extract_hiddens(
    cfg: ExtractionConfig,
    *,
    device: Union[str, torch.device] = "cpu",
    rank: int = 0,
    split: str,
    world_size: int = 1,
) -> Iterable[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states.

    This is a lightweight, functional version of the `Extractor` API.
    """

    # Silence datasets logging messages from all but the first process
    if rank != 0:
        logging.disable(logging.CRITICAL)

    prompt_ds = PromptDataset(cfg.prompts, rank, world_size, split)
    if rank == 0:
        prompt_names = prompt_ds.prompter.all_template_names
        if cfg.prompts.num_variants >= 1:
            print(
                f"Using {cfg.prompts.num_variants} prompts per example: {prompt_names}"
            )
        elif cfg.prompts.num_variants == -1:
            print(f"Using all prompts per example: {prompt_names}")
        else:
            raise ValueError(f"Invalid prompt num_variants: {cfg.prompts.num_variants}")

    model_cls = get_model_class(cfg.model)
    model = assert_type(
        PreTrainedModel, model_cls.from_pretrained(cfg.model, torch_dtype="auto")
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, truncation_side="left", verbose=False
    )
    if cfg.use_encoder_states and not model.config.is_encoder_decoder:
        raise ValueError(
            "use_encoder_states is only compatible with encoder-decoder models."
        )

    num_choices = prompt_ds.num_classes
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Whether to concatenate the question and answer before passing to the model.
    # If False pass them to the encoder and decoder separately.
    is_enc_dec = model.config.is_encoder_decoder
    should_concat = not is_enc_dec or cfg.use_encoder_states

    # If this is an encoder-decoder model and we're passing the answer to the encoder,
    # we don't need to run the decoder at all. Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    if is_enc_dec and cfg.use_encoder_states:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = assert_type(PreTrainedModel, model.get_encoder())

    # Iterating over questions
    layer_indices = cfg.layers or tuple(range(model.config.num_hidden_layers))
    for prompts in prompt_ds:
        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                prompt_ds.num_variants,
                num_choices,
                model.config.hidden_size,
                device=device,
                dtype=torch.int16,
            )
            for layer_idx in layer_indices
        }
        log_probs = torch.empty(
            prompt_ds.num_variants,
            num_choices,
            device=device,
            dtype=torch.float32,
        )
        variant_ids = [prompt.template_name for prompt in prompts]

        # Iterate over variants
        for i, prompt in enumerate(prompts):
            # Iterate over answers
            for j in range(num_choices):
                q = prompt.question
                a = prompt.answers[j]

                x = f"{q} {a}" if should_concat else q
                inputs = tokenizer(
                    x,
                    text_target=[a] if not should_concat else None,  # type: ignore
                    padding=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=True,
                )

                offsets = inputs.pop("offset_mapping").squeeze().tolist()
                outputs = model(**inputs.to(device), output_hidden_states=True)

                if should_concat:
                    # Locate the start of the answer in the tokenized input
                    answer_start = bisect_left(offsets, x.rindex(a), key=itemgetter(1))

                    log_dist = outputs.logits[:, answer_start - 1 : -1].log_softmax(
                        dim=-1
                    )
                    tokens = inputs.input_ids[:, answer_start:]
                else:
                    log_dist = outputs.logits.log_softmax(dim=-1)
                    tokens = inputs.input_ids

                log_probs[i, j] = log_dist.gather(-1, tokens[..., None]).sum()
                hiddens = (
                    outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                )
                # First element of list is the input embeddings
                hiddens = hiddens[1:]

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

        yield dict(
            label=prompts[0].label,
            model_preds=log_probs.softmax(dim=-1),
            variant_ids=variant_ids,
            **hidden_dict,
        )


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def extract(cfg: ExtractionConfig, max_gpus: int = -1) -> DatasetDict:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""

    def get_splits() -> SplitDict:
        available_splits = assert_type(SplitDict, info.splits)
        priorities = {
            Split.TRAIN: 0,
            Split.VALIDATION: 1,
            Split.TEST: 2,
        }
        splits = sorted(available_splits, key=lambda k: priorities.get(k, 100))
        assert len(splits) >= 2, "Must have train and val/test splits"

        # Take the first two splits
        splits = splits[:2]
        print(f"Using '{splits[0]}' for training and '{splits[1]}' for validation")

        # Empty list means no limit
        limit_list = cfg.prompts.max_examples
        if not limit_list:
            limit_list = [int(1e100)]

        # Broadcast the limit to all splits
        if len(limit_list) == 1:
            limit_list *= len(splits)

        limit = {k: v for k, v in zip(splits, limit_list)}
        return SplitDict(
            {
                k: SplitInfo(
                    name=k,
                    num_examples=min(limit[k], v.num_examples),
                    dataset_name=v.dataset_name,
                )
                for k, v in available_splits.items()
                if k in splits
            },
            dataset_name=available_splits.dataset_name,
        )

    model_cfg = AutoConfig.from_pretrained(cfg.model)
    num_variants = cfg.prompts.num_variants
    ds_name, _, config_name = cfg.prompts.dataset.partition(" ")
    info = get_dataset_config_info(ds_name, config_name or None)

    features = assert_type(Features, info.features)
    label_col = cfg.prompts.label_column or infer_label_column(features)

    splits = get_splits()

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
        "label": features[label_col],
        "model_preds": Array2D(
            dtype="float32",
            shape=(num_variants, 2),
        ),
    }
    devices = select_usable_devices(max_gpus)
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
                split=[split_name] * len(devices),
                world_size=[len(devices)] * len(devices),
            ),
        )
        for (split_name, split_info) in splits.items()
    }

    ds = dict()
    for split, builder in builders.items():
        builder.download_and_prepare(num_proc=len(devices))
        ds[split] = builder.as_dataset(split=split)
    return DatasetDict(ds)
