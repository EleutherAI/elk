"""Functions for extracting the hidden states of a model."""

from ..utils import pytree_map
from .prompt_collator import Prompt, PromptCollator
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
from typing import cast, Iterable, Literal, Sequence
import torch
import torch.distributed as dist


@torch.autocast("cuda", enabled=torch.cuda.is_available())  # type: ignore
@torch.no_grad()
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
) -> Iterable[tuple[torch.Tensor, list[int]]]:
    """Run inference on a model with a set of prompts, yielding the hidden states.

    Args:
        model: The model to run inference on.
        tokenizer: The tokenizer to use for tokenization.
        collator: The PromptCollator to use for generating prompts.
        batch_size: The batch size to use for inference.
        layers (Sequence[int]): The layers to extract hidden states from.
        prompt_suffix (str): A string to append to the end of each prompt.
        token_loc: The location of the token to extract hidden states from.
            can be either "first", "last", or "mean". Defaults to "last".
        use_encoder_states: Whether to use the encoder states instead of the
            decoder states. This allows simplification from an encoder-decoder
            model to an encoder-only model. Defaults to False.
    """

    device = model.device
    num_choices = len(collator.labels)

    # TODO: Make this configurable or something
    # Token used to separate the question from the answer
    sep_token = tokenizer.sep_token or "\n"

    # TODO: Maybe also make this configurable?
    # We want to make sure the answer is never truncated
    tokenizer.truncation_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(strings: list[str]):
        return pytree_map(
            lambda x: x.to(device),
            tokenizer(
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
            prompt.to_string(i, sep=sep_token) + prompt_suffix
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
                prompt.answers[i] + prompt_suffix
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

        if token_loc == "first":
            hiddens = [h[..., 0, :] for h in hiddens]
        elif token_loc == "last":
            # Because of padding, the last token is going to be at a different index
            # for each example, so we use gather.
            B, C, _, D = hiddens[0].shape
            lengths = attention_mask.sum(dim=-1).view(B, C, 1, 1)
            indices = lengths.sub(1).expand(B, C, 1, D)
            hiddens = [h.gather(index=indices, dim=-2).squeeze(-2) for h in hiddens]
        elif token_loc == "mean":
            hiddens = [h.mean(dim=-2) for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {token_loc}")

        if layers:
            hiddens = [hiddens[i] for i in layers]

        # [batch size, layers, num choices, hidden size]
        return torch.stack(hiddens, dim=1)

    # If this is an encoder-decoder model and we're passing the answer to the encoder,
    # we don't need to run the decoder at all. Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    is_enc_dec = model.config.is_encoder_decoder
    if is_enc_dec and use_encoder_states:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = cast(PreTrainedModel, model.get_encoder())

    # Whether to concatenate the question and answer before passing to the model.
    # If False pass them to the encoder and decoder separately.
    should_concat = not is_enc_dec or use_encoder_states

    dl = DataLoader(
        collator,
        batch_size=batch_size,
        collate_fn=collate if should_concat else collate_enc_dec,
    )

    # Iterating over questions
    rank = dist.get_rank() if dist.is_initialized() else 0
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
            yield torch.stack(outputs.decoder_hidden_states, dim=2), labels

        # Condition 2: Either a decoder-only transformer or a transformer encoder
        else:
            choices, labels = batch

            # Skip the input embeddings which are unlikely to be interesting
            h = model(**choices, output_hidden_states=True).hidden_states[1:]
            yield reduce_seqs(h, choices["attention_mask"]), labels
