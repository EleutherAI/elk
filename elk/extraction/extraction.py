"""Functions for extracting the hidden states of a model."""

from ..utils import pytree_map
from .prompt_collator import Prompt, PromptCollator
from dataclasses import dataclass
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
from typing import cast, Literal, Iterator, Sequence
import numpy as np
import torch
import torch.multiprocessing as mp


@dataclass
class ExtractionParameters:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    collator: PromptCollator
    batch_size: int = 1
    layers: Sequence[int] = ()
    prompt_suffix: str = ""
    token_loc: Literal["first", "last", "mean"] = "last"
    use_encoder_states: bool = False


def extract_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    collator: PromptCollator,
    *,
    # TODO: Bring back auto-batching when we have a good way to prevent excess padding
    batch_size: int = 1,
    layers: Sequence[int] = (),
    prompt_suffix: str = "",
    token_loc: Literal["first", "last", "mean"] = "last",
    use_encoder_states: bool = False,
    seed_start: int = 42,
):
    """Run inference on a model with a set of prompts, yielding the hidden states."""

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    # use different random seed for each process
    curr_seed = seed_start
    workers = []

    # Start the workers.
    num_gpus = torch.cuda.device_count()
    shards = np.array_split(np.arange(len(collator)), num_gpus)
    for rank, proc_indices in enumerate(shards):
        params = ExtractionParameters(
            model=model,
            tokenizer=tokenizer,
            collator=collator.split_and_copy(proc_indices, curr_seed),
            batch_size=batch_size,
            layers=layers,
            prompt_suffix=prompt_suffix,
            token_loc=token_loc,
            use_encoder_states=use_encoder_states,
        )

        worker = ctx.Process(
            target=_extract_hiddens_process,
            args=(queue, params, rank),
        )
        worker.start()
        workers.append(worker)

        curr_seed += 1

    # Consume the results from the queue
    for _ in range(len(collator)):
        yield queue.get()

    # Clean up
    for worker in workers:
        worker.join()


@torch.no_grad()
def _extract_hiddens_process(
    queue: mp.Queue,
    params: ExtractionParameters,
    rank: int,
) -> Iterator[dict]:
    """
    Do inference on a model with a set of prompts on a single process.
    To be passed to Dataset.from_generator.
    """
    print(f"Process with rank={rank}")

    model = params.model.to(f"cuda:{rank}")
    num_choices = len(params.collator.labels)

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
    def collate(prompts: list[Prompt]) -> tuple[BatchEncoding, list[int]]:
        choices = [
            prompt.to_string(i, sep=sep_token) + params.prompt_suffix
            for prompt in prompts
            for i in range(num_choices)
        ]
        return tokenize(choices), [prompt.label for prompt in prompts]

    def collate_enc_dec(
        prompts: list[Prompt],
    ) -> tuple[BatchEncoding, BatchEncoding, list[int]]:
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
        return tokenized_questions, tokenized_answers, labels

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
    for batch in tqdm(dl, position=rank):
        # Condition 1: Encoder-decoder transformer, with answer in the decoder
        if not should_concat:
            questions, answers, labels = batch
            outputs = model(
                **questions,
                **{f"decoder_{k}": v for k, v in answers.items()},
                output_hidden_states=True,
            )
            # [batch_size, num_layers, num_choices, hidden_size]
            yield {
                "hiddens": torch.stack(outputs.decoder_hidden_states, dim=2),
                "labels": labels,
            }

        # Condition 2: Either a decoder-only transformer or a transformer encoder
        else:
            choices, labels = batch

            # Skip the input embeddings which are unlikely to be interesting
            h = model(**choices, output_hidden_states=True).hidden_states[1:]
            queue.put(
                {
                    "hiddens": reduce_seqs(h, choices["attention_mask"]),
                    "labels": labels,
                }
            )
