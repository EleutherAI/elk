from ..utils import pytree_map
from .prompt_collator import Prompt, PromptCollator
from accelerate import find_executable_batch_size
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import cast, Iterable, Sequence
import torch


# We use this function to find where the answer starts in the tokenized prompt.
# This way, we're robust to idiosyncrasies in the tokenizer.
def common_prefix_len(*seqs: Sequence) -> int:
    """Compute the length of the common prefix of N sequences."""
    for i, elems in enumerate(zip(*seqs)):
        pivot, *rest = elems
        if not all(elem == pivot for elem in rest):
            return i

    return min(len(x) for x in seqs)


# TODO: Add support for batched inference
@torch.autocast("cuda", enabled=torch.cuda.is_available())
@torch.no_grad()
def extract_hiddens(
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    collator: PromptCollator,
) -> Iterable[tuple[torch.Tensor, list[int]]]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""
    yield from _extract_inner(args, model, tokenizer, collator)  # type: ignore


@find_executable_batch_size(starting_batch_size=32)
def _extract_inner(
    batch_size: int,
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    collator: PromptCollator,
):
    print(f"Using batch size: {batch_size}")
    num_choices = len(collator.labels)

    # We want to make sure the answer is never truncated
    tokenizer.truncation_side = "left"

    # This is sort of annoying. We have a list of questions, and for each data point we
    # have K different answers. We want to run inference on each question-answer pair,
    # but we want to batch them together so that we can run inference on the whole
    # batch. So we have to do some gymnastics to get the right shape.
    # This function returns the flattened questions and answers, and the labels for
    # each question-answer pair. After inference we need to reshape the results.
    def collate(items: list[tuple[list[Prompt], int]]) -> tuple[dict, list[int]]:
        nested_choices = [
            [str(choice) + args.prompt_suffix for choice in choices]
            for choices, _ in items
        ]
        tokenized = pytree_map(
            # Unflatten for inference
            lambda x: x.to(args.device),
            tokenizer(
                # Flatten to fit in the tokenizer
                [choice for choices in nested_choices for choice in choices],
                padding=True,
                return_tensors="pt",
                truncation=True,
            ),
        )
        return tokenized, [label for _, label in items]

    def reduce_seqs(
        hiddens: list[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce sequences of hiddens into single vectors."""

        # Unflatten the hiddens
        hiddens = [rearrange(h, "(b n) l d -> b n l d", n=num_choices) for h in hiddens]

        if args.token_loc == "first":
            hiddens = [h[..., 0, :].squeeze() for h in hiddens]
        elif args.token_loc == "last":
            # Because of padding, the last token is going to be at a different index
            # for each example, so we use gather.
            B, C, *_ = hiddens[0].shape
            lengths = attention_mask.sum(dim=-1).view(B, C, 1, 1)
            hiddens = [h.gather(index=lengths - 1, dim=-2).squeeze() for h in hiddens]
        elif args.token_loc == "mean":
            hiddens = [h.mean(dim=-2).squeeze() for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {args.token_loc}")

        if args.layers is not None:
            hiddens = [hiddens[i] for i in args.layers]

        # [batch size, layers, num choices, hidden size]
        return torch.stack(hiddens, dim=1)

    def tokenize(s: list[str]):
        return pytree_map(
            lambda x: x.to(args.device),
            tokenizer(s, return_tensors="pt", truncation=True, padding=True),
        )

    is_enc_dec = model.config.is_encoder_decoder

    # If this is an encoder-decoder model and we're passing the answer to the encoder,
    # we don't need to run the decoder at all. Just strip it off, making the problem
    # equivalent to a regular encoder-only model.
    if is_enc_dec and args.use_encoder_states:
        # This isn't actually *guaranteed* by HF, but it's true for all existing models
        if not hasattr(model, "get_encoder") or not callable(model.get_encoder):
            raise ValueError(
                "Encoder-decoder model doesn't have expected get_encoder() method"
            )

        model = cast(PreTrainedModel, model.get_encoder())

    dl = DataLoader(collator, batch_size=batch_size, collate_fn=collate)

    # Iterating over questions
    for choices, labels in tqdm(dl):
        # There are three conditions here:
        # 1) Encoder-decoder transformer, with answer in the decoder
        # 2) Decoder-only transformer
        # 3) Transformer encoder
        # In cases 1 & 2, we can reuse hidden states for the question.
        # In case 3, we have to recompute all hidden states every time.

        # Condition 1: Encoder-decoder transformer, with answer in the decoder
        if is_enc_dec and not args.use_encoder_states:
            prompt0, *rest = choices

            # First run the full model on the question + answer 0
            output0 = model(
                **tokenize(prompt0.question),
                **{
                    f"decoder_{k}": v
                    for k, v in tokenize(prompt0.answer + args.prompt_suffix).items()
                },
                output_hidden_states=True,
            )
            # Then run the decoder on the other answers with cached encoder states
            hiddens = [reduce_seqs(output0.decoder_hidden_states)] + [
                reduce_seqs(
                    model(
                        encoder_hidden_states=output0.encoder_hidden_states,
                        labels=tokenize(prompt.answer + args.prompt_suffix),
                        output_hidden_states=True,
                    ).decoder_hidden_states
                )
                for prompt in rest
            ]
            # [batch_size, num_layers, num_choices, hidden_size]
            yield torch.stack(hiddens, dim=2), labels

        # Either a decoder-only transformer or a transformer encoder
        else:
            h = model(**choices, output_hidden_states=True).hidden_states
            yield reduce_seqs(h, choices["attention_mask"]), labels
