"""Functions for extracting the hidden states of a model."""

from .prompt_dataset import Prompt, PromptDataset, PromptConfig
from ..utils import select_usable_gpus
from dataclasses import dataclass, InitVar
from datasets import Array3D, Features, Dataset, DatasetDict, Sequence, Value
from simple_parsing.helpers import field, Serializable
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)
from typing import cast, Literal, Iterable
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np


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
            config = AutoConfig.from_pretrained(self.model)
            assert isinstance(config, PretrainedConfig)

            self.layers = tuple(range(0, config.num_hidden_layers, layer_stride))


def extract_to_disk(cfg: ExtractionConfig, output_path: Path):
    if not output_path.exists():
        output_path.mkdir()

    extract_to_dataset(cfg).save_to_disk(output_path)


def extract_hiddens(
    cfg: ExtractionConfig,
    *,
    # TODO: Bring back auto-batching when we have a good way to prevent excess padding
    batch_size: int = 1,
    max_gpus: int = -1,
    split: str,
) -> Iterable[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    # TODO: Use a heuristic based on params to determine minimum VRAM
    gpu_indices = select_usable_gpus(max_gpus)
    num_gpus = len(gpu_indices)

    # Spawn a process for each GPU
    ctx = torch.multiprocessing.spawn(
        _extract_hiddens_process,
        args=(gpu_indices, queue, cfg, batch_size, split),
        nprocs=num_gpus,
        join=False,
    )
    assert ctx is not None

    # Yield results from the queue
    procs_running = num_gpus
    while procs_running > 0:
        output = queue.get()

        # None is a sentinel value indicating that a process has finished
        if output is None:
            procs_running -= 1
        else:
            assert isinstance(output, dict)
            yield output

    # Clean up
    ctx.join()


def extract_to_dataset(
    cfg: ExtractionConfig,
    max_gpus: int = -1,
) -> DatasetDict:
    config = AutoConfig.from_pretrained(cfg.model)
    num_variants = cfg.prompts.num_variants

    layer_cols = {
        f"hidden_{layer}": Array3D(
            dtype="int16",
            shape=(num_variants, 2, config.hidden_size),
        )
        for layer in cfg.layers or range(config.num_hidden_layers)
    }
    other_cols = {
        "variant_ids": Sequence(
            Value(dtype="string"),
            length=num_variants,
        ),
        "predicate_id": Value("int32"),
        "label": Value("int32"),
        "example_input": Value(
            "string"
        ),  # exact input to the LM for one vairant+answer
    }

    return DatasetDict(
        {
            split_name: Dataset.from_generator(
                extract_hiddens,
                gen_kwargs=dict(cfg=cfg, split=split_name),
                features=Features({**layer_cols, **other_cols}),
            )
            for split_name in ["train", "validation"]
        }
    )


@torch.no_grad()
def _extract_hiddens_process(
    rank: int,
    gpu_indices: list[int],
    queue: mp.Queue,
    cfg: ExtractionConfig,
    batch_size: int,
    split: str,
):
    """
    Do inference on a model with a set of prompts on a single process.
    To be passed to Dataset.from_generator.
    """
    local_gpu = gpu_indices[rank]
    world_size = len(gpu_indices)

    prompts_dataset = PromptDataset(cfg.prompts, rank, world_size, split)
    if rank == 0:
        prompt_names = prompts_dataset.prompter.all_template_names
        if cfg.prompts.num_variants >= 1:
            print(
                f"Using {cfg.prompts.num_variants} prompts per example: {prompt_names}"
            )
        elif cfg.prompts.num_variants == -1:
            print(f"Using all prompts per example: {prompt_names}")
        else:
            raise ValueError(f"Invalid prompt num_variants: {cfg.prompts.num_variants}")
    else:
        logging.getLogger("transformers").setLevel(logging.CRITICAL)

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    model = AutoModel.from_pretrained(cfg.model, torch_dtype="auto").to(
        f"cuda:{local_gpu}"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    if cfg.use_encoder_states and not model.config.is_encoder_decoder:
        raise ValueError(
            "use_encoder_states is only compatible with encoder-decoder models."
        )

    # TODO: Make this configurable or something
    # Token used to separate the question from the answer
    num_choices = prompts_dataset.num_classes
    sep_token = tokenizer.sep_token or "\n"

    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    tokenizer.truncation_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(strings: list[str], **kwargs):
        return tokenizer(
            strings,
            padding=True,
            return_tensors="pt",
            truncation=True,
            **kwargs,
        ).to(f"cuda:{local_gpu}")

    # This function returns the flattened questions and answers. After inference we
    # need to reshape the results.
    def collate(
        prompts_batch: list[list[Prompt]],
    ) -> tuple[list[list[BatchEncoding]], list[Prompt], str]:
        assert len(prompts_batch) == 1  # We're not batching
        prompts = prompts_batch[0]
        return (
            [
                [
                    tokenize([prompt.to_string(i, sep=sep_token)])
                    for i in range(num_choices)
                ]
                for prompt in prompts
            ],
            prompts,
            prompts[0].to_string(0, sep=sep_token),
        )

    def collate_enc_dec(
        prompts_batch: list[list[Prompt]],
    ) -> tuple[list[list[BatchEncoding]], list[Prompt], str]:
        assert len(prompts_batch) == 1  # We're not batching
        prompts = prompts_batch[0]
        return (
            [
                [
                    tokenize([prompt.question], text_target=[target])
                    for target in prompt.answers
                ]
                for prompt in prompts
            ],
            prompts,
            prompts[0].question + sep_token + prompts[0].answers[0],
        )

    # If this is an encoder-decoder model and we're passing the answer to the encoder,
    # we don't need to run the decoder at all. Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    is_enc_dec = model.config.is_encoder_decoder
    if is_enc_dec and cfg.use_encoder_states:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = cast(PreTrainedModel, model.get_encoder())

    # Whether to concatenate the question and answer before passing to the model.
    # If False pass them to the encoder and decoder separately.
    should_concat = not is_enc_dec or cfg.use_encoder_states
    dl = DataLoader(
        prompts_dataset,
        batch_size=1,
        collate_fn=collate if should_concat else collate_enc_dec,
    )

    # Iterating over questions
    for batch in dl:
        inputs, prompts, example_text = batch

        hidden_dict = {
            f"hidden_{layer_idx}": np.empty(
                (prompts_dataset.num_variants, num_choices, model.config.hidden_size),
                dtype=np.int16,
            )
            for layer_idx in cfg.layers or range(model.config.num_hidden_layers)
        }

        variant_ids = []
        # Iterate over variants
        for variant_index, (variant_inputs, prompt) in enumerate(zip(inputs, prompts)):
            # Iterate over answers
            for answer_index, inpt in enumerate(variant_inputs):
                outputs = model(**inpt, output_hidden_states=True)

                raw_hiddens = (
                    outputs.get("decoder_hidden_states") or outputs["hidden_states"]
                )
                hiddens = raw_hiddens[
                    1:
                ]  # First element of tuple is the input embeddings

                # Throw out layers we don't care about
                if cfg.layers:
                    hiddens = [hiddens[i] for i in cfg.layers]

                # Current shape of each element: (batch_size, seq_len, hidden_size)
                if cfg.token_loc == "first":
                    hiddens = [h[..., 0, :] for h in hiddens]
                elif cfg.token_loc == "last":
                    hiddens = [h[..., -1, :] for h in hiddens]
                elif cfg.token_loc == "mean":
                    hiddens = [h.mean(dim=-2) for h in hiddens]
                else:
                    raise ValueError(f"Invalid token_loc: {cfg.token_loc}")

                for layer_idx, hidden in zip(
                    cfg.layers or range(model.config.num_hidden_layers), hiddens
                ):
                    int_hidden = (
                        hidden.cpu().numpy().astype(np.float16).view(dtype=np.int16)
                    )
                    hidden_dict[f"hidden_{layer_idx}"][
                        variant_index, answer_index
                    ] = int_hidden
            variant_ids.append(prompt.template_name)

        hidden_dict["variant_ids"] = variant_ids  # type: ignore
        hidden_dict["example_input"] = example_text
        hidden_dict["label"] = prompt.label  # type: ignore
        hidden_dict["predicate_id"] = prompt.predicate_id  # type: ignore
        queue.put(hidden_dict)

    # Signal to the consumer that we're done
    queue.put(None)
