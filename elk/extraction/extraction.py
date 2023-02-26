"""Functions for extracting the hidden states of a model."""

from .prompt_dataset import Prompt, PromptDataset, PromptConfig
from ..utils import assert_type, select_usable_gpus
from dataclasses import dataclass, InitVar
from datasets import (
    Array3D,
    DatasetDict,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    get_dataset_config_info,
    Sequence,
    Split,
    SplitDict,
    SplitGenerator,
    SplitInfo,
    Value,
)
from simple_parsing.helpers import field, Serializable
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
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

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    model = AutoModel.from_pretrained(cfg.model, torch_dtype="auto").to(device)
    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, truncation_side="left")

    if cfg.use_encoder_states and not model.config.is_encoder_decoder:
        raise ValueError(
            "use_encoder_states is only compatible with encoder-decoder models."
        )

    # TODO: Make this configurable or something
    # Token used to separate the question from the answer
    num_choices = prompt_ds.num_classes
    sep_token = tokenizer.sep_token or "\n"

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Whether to concatenate the question and answer before passing to the model.
    # If False pass them to the encoder and decoder separately.
    is_enc_dec = model.config.is_encoder_decoder
    should_concat = not is_enc_dec or cfg.use_encoder_states

    def tokenize(prompt: Prompt, idx: int, **kwargs):
        return tokenizer(
            (
                [prompt.to_string(idx, sep=sep_token)]
                if should_concat
                else [prompt.question]
            ),
            text_target=[prompt.answers[idx]] if not should_concat else None,
            padding=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        ).to(device)

    # This function returns the flattened questions and answers. After inference we
    # need to reshape the results.
    def collate(prompts: list[Prompt]) -> list[list[BatchEncoding]]:
        return [[tokenize(prompt, i) for i in range(num_choices)] for prompt in prompts]

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
        inputs = collate(prompts)
        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                prompt_ds.num_variants,
                num_choices,
                model.config.hidden_size,
                device=device,
            )
            for layer_idx in layer_indices
        }
        variant_ids = [prompt.template_name for prompt in prompts]

        # Iterate over variants
        for i, variant_inputs in enumerate(inputs):
            # Iterate over answers
            for j, inpt in enumerate(variant_inputs):
                outputs = model(**inpt, output_hidden_states=True)

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
                    hidden_dict[f"hidden_{layer_idx}"][i, j] = hidden

        yield dict(
            label=prompts[0].label,
            variant_ids=variant_ids,
            **hidden_dict,
        )


class Extractor(GeneratorBasedBuilder):
    """Builds a HuggingFace dataset containing hidden states extracted from a model."""

    def __init__(self, cfg: ExtractionConfig, max_gpus: int = -1, **kwargs):
        # Fetch metadata about the dataset without necessarily downloading it yet
        ds_name, _, config_name = cfg.prompts.dataset.partition(" ")
        info = get_dataset_config_info(ds_name, config_name or None)

        self.base_info = info
        self.cfg = cfg

        # TODO: Use a heuristic based on params to determine minimum VRAM
        self.gpus = select_usable_gpus(max_gpus)

        super().__init__(**kwargs)

    def extract(self) -> DatasetDict:
        """Extract hidden states and return a `DatasetDict` containing them."""
        # "Download" is a misnomer here. We're running inference on a dataset and
        # gathering the hidden states.
        self.download_and_prepare(num_proc=len(self.gpus))

        return DatasetDict(
            {split: self.as_dataset(split=Split(split)) for split in self.splits}
        )

    @property
    def splits(self) -> SplitDict:
        """Return the standard splits that are available in the dataset."""
        base_splits = assert_type(SplitDict, self.base_info.splits)
        splits = set(base_splits) & {Split.TRAIN, Split.VALIDATION, Split.TEST}

        # If we're using the validation set, we need to remove the test set
        if Split.VALIDATION in splits and Split.TEST in splits:
            splits.remove(Split.TEST)

        limit = self.cfg.prompts.max_examples or int(1e100)
        return SplitDict(
            {
                k: SplitInfo(
                    name=k,
                    num_examples=min(limit, v.num_examples),
                    dataset_name=v.dataset_name,
                )
                for k, v in base_splits.items()
                if k in splits
            },
            dataset_name=base_splits.dataset_name,
        )

    def _info(self) -> DatasetInfo:
        model_cfg = AutoConfig.from_pretrained(self.cfg.model)
        num_variants = self.cfg.prompts.num_variants

        layer_cols = {
            f"hidden_{layer}": Array3D(
                dtype="float32",
                shape=(num_variants, 2, model_cfg.hidden_size),
            )
            for layer in self.cfg.layers or range(model_cfg.num_hidden_layers)
        }
        other_cols = {
            "variant_ids": Sequence(
                Value(dtype="string"),
                length=num_variants,
            ),
            "label": Value("int32"),
        }

        return DatasetInfo(
            description=f"Hiddens for {self.cfg.model} on {self.cfg.prompts.dataset}",
            features=Features({**layer_cols, **other_cols}),
            citation=self.base_info.citation,
            splits=self.splits,
        )

    def _split_generators(self, _) -> list[SplitGenerator]:
        return [
            SplitGenerator(
                name=split,
                gen_kwargs=dict(
                    cfg=[self.cfg] * len(self.gpus),
                    device=[f"cuda:{i}" for i in self.gpus],
                    rank=list(range(len(self.gpus))),
                    split=[split] * len(self.gpus),
                    world_size=[len(self.gpus)] * len(self.gpus),
                ),
            )
            for split in self.splits
        ]

    def _generate_examples(self, **gen_kwargs) -> Iterable[tuple[int, dict]]:
        yield from enumerate(_extraction_worker(**gen_kwargs))


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})
