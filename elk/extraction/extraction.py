from ..utils import pytree_map
from .prompt_collator import Prompt, PromptCollator
from accelerate import find_executable_batch_size
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import cast, Iterable
import torch


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


# TODO: Bring back batching when we have a good way to prevent excess padding
@find_executable_batch_size(starting_batch_size=1)
def _extract_inner(
    batch_size: int,
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    collator: PromptCollator,
):
    print(f"Using batch size: {batch_size}")
    num_choices = len(collator.labels)

    # TODO: Make this configurable or something
    # We want to make sure the answer is never truncated
    tokenizer.truncation_side = "left"

    # This function returns the flattened questions and answers, and the labels for
    # each question-answer pair. After inference we need to reshape the results.
    def collate(prompts: list[Prompt]) -> tuple[dict, list[int]]:
        choices = [
            prompt.to_string(i) + args.prompt_suffix
            for prompt in prompts
            for i in range(num_choices)
        ]
        tokenized = pytree_map(
            # Unflatten for inference
            lambda x: x.to(args.device),
            tokenizer(
                choices,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ),
        )
        return tokenized, [prompt.label for prompt in prompts]

    def reduce_seqs(
        hiddens: list[torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce sequences of hiddens into single vectors."""

        # Unflatten the hiddens
        hiddens = [rearrange(h, "(b c) l d -> b c l d", c=num_choices) for h in hiddens]

        if args.token_loc == "first":
            hiddens = [h[..., 0, :] for h in hiddens]
        elif args.token_loc == "last":
            # Because of padding, the last token is going to be at a different index
            # for each example, so we use gather.
            B, C, _, D = hiddens[0].shape
            lengths = attention_mask.sum(dim=-1).view(B, C, 1, 1)
            indices = lengths.sub(1).expand(B, C, 1, D)
            hiddens = [h.gather(index=indices, dim=-2).squeeze(-2) for h in hiddens]
        elif args.token_loc == "mean":
            hiddens = [h.mean(dim=-2) for h in hiddens]
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
            hiddens = [reduce_seqs(output0.decoder_hidden_states[1:])] + [
                reduce_seqs(
                    model(
                        encoder_hidden_states=output0.encoder_hidden_states[1:],
                        labels=tokenize(prompt.answer + args.prompt_suffix),
                        output_hidden_states=True,
                    ).decoder_hidden_states[1:]
                )
                for prompt in rest
            ]
            # [batch_size, num_layers, num_choices, hidden_size]
            yield torch.stack(hiddens, dim=2), labels

        # Either a decoder-only transformer or a transformer encoder
        else:
            # Skip the input embeddings which are unlikely to be interesting
            h = model(**choices, output_hidden_states=True).hidden_states[1:]
            yield reduce_seqs(h, choices["attention_mask"]), labels
