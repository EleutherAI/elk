"""Functions for extracting the hidden states of a model."""

from ..utils import pytree_map
from .prompt_collator import Prompt, PromptCollator
from dataclasses import dataclass
from einops import rearrange
from torch.utils.data import DataLoader
from transformers import (
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModel,
)
from typing import cast, Literal, Sequence
import logging
import numpy as np
import torch
import torch.multiprocessing as mp


@dataclass
class ExtractionParameters:
    model_str: str
    tokenizer: PreTrainedTokenizerBase
    collator: PromptCollator
    batch_size: int = 1
    layers: Sequence[int] = ()
    prompt_suffix: str = ""
    token_loc: Literal["first", "last", "mean"] = "last"
    use_encoder_states: bool = False


def extract_hiddens(
    model_str: str,
    tokenizer: PreTrainedTokenizerBase,
    collator: PromptCollator,
    *,
    # TODO: Bring back auto-batching when we have a good way to prevent excess padding
    batch_size: int = 1,
    layers: Sequence[int] = (),
    prompt_suffix: str = "",
    token_loc: Literal["first", "last", "mean"] = "last",
    use_encoder_states: bool = False,
):
    """Run inference on a model with a set of prompts, yielding the hidden states."""

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    num_gpus = torch.cuda.device_count()
    params = ExtractionParameters(
        model_str=model_str,
        tokenizer=tokenizer,
        collator=collator,
        batch_size=batch_size,
        layers=layers,
        prompt_suffix=prompt_suffix,
        token_loc=token_loc,
        use_encoder_states=use_encoder_states,
    )

    # Spawn a process for each GPU
    ctx = torch.multiprocessing.spawn(
        _extract_hiddens_process,
        args=(num_gpus, queue, params),
        nprocs=num_gpus,
        join=False,
    )
    assert ctx is not None

    # Yield results from the queue
    for _ in range(len(collator)):
        yield queue.get()

    # Clean up
    ctx.join()


@torch.no_grad()
def _extract_hiddens_process(
    rank: int,
    world_size: int,
    queue: mp.Queue,
    params: ExtractionParameters,
):
    """
    Do inference on a model with a set of prompts on a single process.
    To be passed to Dataset.from_generator.
    """
    print(f"Process with rank={rank}")
    if rank != 0:
        logging.getLogger("transformers").setLevel(logging.CRITICAL)

    num_choices = params.collator.num_classes
    shards = np.array_split(np.arange(len(params.collator)), world_size)
    params.collator.select_(shards[rank])

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    model = AutoModel.from_pretrained(params.model_str, torch_dtype="auto").to(
        f"cuda:{rank}"
    )

    if params.use_encoder_states and not model.config.is_encoder_decoder:
        raise ValueError(
            "use_encoder_states is only compatible with encoder-decoder models."
        )

    # TODO: Make this configurable or something
    # Token used to separate the question from the answer
    sep_token = params.tokenizer.sep_token or "\n"

    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    params.tokenizer.truncation_side = "left"
    if not params.tokenizer.pad_token:
        params.tokenizer.pad_token = params.tokenizer.eos_token

    def tokenize(strings: list[str]):
        return pytree_map(
            lambda x: x.to(f"cuda:{rank}"),
            params.tokenizer(
                strings,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ),
        )

    # This function returns the flattened questions and answers, and the labels for
    # each question-answer pair. After inference we need to reshape the results.
    def collate(prompts: list[Prompt]) -> tuple[BatchEncoding, list[int], list[Prompt]]:
        choices = [
            prompt.to_string(i, sep=sep_token) + params.prompt_suffix
            for prompt in prompts
            for i in range(num_choices)
        ]
        return tokenize(choices), [prompt.label for prompt in prompts], prompts

    def collate_enc_dec(
        prompts: list[Prompt],
    ) -> tuple[BatchEncoding, BatchEncoding, list[int], list[Prompt]]:
        tokenized_questions = tokenize(
            [prompt.question for prompt in prompts for _ in range(num_choices)]
        )
        tokenized_answers = tokenize(
            [
                prompt.answers[i] + params.prompt_suffix
                for prompt in prompts
                for i in range(num_choices)
            ]
        )
        labels = [prompt.label for prompt in prompts]
        return tokenized_questions, tokenized_answers, labels, prompts

    def reduce_seqs(
        hiddens: list[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce sequences of hiddens into single vectors."""

        # Unflatten the hiddens
        hiddens = [rearrange(h, "(b c) l d -> b c l d", c=num_choices) for h in hiddens]

        if params.token_loc == "first":
            hiddens = [h[..., 0, :] for h in hiddens]
        elif params.token_loc == "last":
            # Because of padding, the last token is going to be at a different index
            # for each example, so we use gather.
            B, C, _, D = hiddens[0].shape
            lengths = attention_mask.sum(dim=-1).view(B, C, 1, 1)
            indices = lengths.sub(1).expand(B, C, 1, D)
            hiddens = [h.gather(index=indices, dim=-2).squeeze(-2) for h in hiddens]
        elif params.token_loc == "mean":
            hiddens = [h.mean(dim=-2) for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {params.token_loc}")

        if params.layers:
            hiddens = [hiddens[i] for i in params.layers]

        # [batch size, layers, num choices, hidden size]
        return torch.stack(hiddens, dim=1)

    # If this is an encoder-decoder model and we're passing the answer to the encoder,
    # we don't need to run the decoder at all. Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    is_enc_dec = model.config.is_encoder_decoder
    if is_enc_dec and params.use_encoder_states:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = cast(PreTrainedModel, model.get_encoder())

    # Whether to concatenate the question and answer before passing to the model.
    # If False pass them to the encoder and decoder separately.
    should_concat = not is_enc_dec or params.use_encoder_states

    dl = DataLoader(
        params.collator,
        batch_size=params.batch_size,
        collate_fn=collate if should_concat else collate_enc_dec,
    )

    # Iterating over questions
    for batch in dl:
        # Condition 1: Encoder-decoder transformer, with answer in the decoder
        if not should_concat:
            questions, answers, labels, prompts = batch
            outputs = model(
                **questions,
                **{f"decoder_{k}": v for k, v in answers.items()},
                output_hidden_states=True,
            )
            # [batch_size, num_layers, num_choices, hidden_size]
            # need to convert hidden states to numpy array first or
            # you get a ConnectionResetErrror
            hiddens = torch.stack(outputs.decoder_hidden_states, dim=2).cpu().numpy()

        # Condition 2: Either a decoder-only transformer or a transformer encoder
        else:
            choices, labels, prompts = batch

            # Skip the input embeddings which are unlikely to be interesting
            h = model(**choices, output_hidden_states=True).hidden_states[1:]

            # need to convert hidden states to numpy array first or
            # you get a ConnectionResetErrror
            hiddens = reduce_seqs(h, choices["attention_mask"]).cpu().numpy()

        # from_generator doesn't deal with batched output, so we split it up here
        for i in range(params.batch_size):
            queue.put(
                {
                    "hiddens": hiddens[i].astype(np.float16),
                    "layers": params.layers,
                    "label": labels[i],
                    "answers": prompts[i].answers,
                    "template_name": prompts[i].template_name,
                    "text": prompts[i].question,
                    "predicate": prompts[i].predicate,
                }
            )
