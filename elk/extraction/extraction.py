from ..utils import pytree_map
from .prompt_collator import PromptCollator
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
) -> Iterable[tuple[torch.Tensor, int]]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""

    def reduce_seqs(hiddens: list[torch.Tensor]) -> torch.Tensor:
        """Reduce sequences of hiddens into single vectors."""
        if args.token_loc == "first":
            hiddens = [h[:, 0].squeeze() for h in hiddens]
        elif args.token_loc == "last":
            hiddens = [h[:, -1].squeeze() for h in hiddens]
        elif args.token_loc == "mean":
            hiddens = [h.mean(dim=1).squeeze() for h in hiddens]
        else:
            raise ValueError(f"Invalid token_loc: {args.token_loc}")

        if args.layer is not None:
            hiddens = [hiddens[args.layer]]

        return torch.stack(hiddens)

    def tokenize(s: str):
        return tokenizer(s, return_tensors="pt").input_ids.to(args.device)

    # We'd like to be able to save compute by reusing hiddens with `past_key_values`
    # when we can, and this requires knowing if the model uses causal masking.
    # This heuristic may have some false negatives, but it should be safe. The HF docs
    # say that the classes in "architectures" should be suitable for this *specific*
    # checkpoint- for example `bert-base-uncased` only lists `BertForMaskedLM`, even
    # though there is a `BertForCausalLM` class.
    is_causal = any(
        arch.endswith("ForCausalLM")
        for arch in getattr(model.config, "architectures", [""])
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

    # Iterating over questions
    for prompts, label in tqdm(collator):
        # There are three conditions here:
        # 1) Encoder-decoder transformer, with answer in the decoder
        # 2) Decoder-only transformer
        # 3) Transformer encoder
        # In cases 1 & 2, we can reuse hidden states for the question.
        # In case 3, we have to recompute all hidden states every time.
        prompt0, *rest = prompts

        # Condition 1: Encoder-decoder transformer, with answer in the decoder
        if is_enc_dec and not args.use_encoder_states:
            # First run the full model on the question + answer 0
            output0 = model(
                input_ids=tokenize(prompt0.question),
                labels=tokenize(prompt0.answer + args.prompt_suffix),
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

        # Either a decoder-only transformer or a transformer encoder
        else:
            # First run the model on the question + answer 0
            tokenized_prompts = [
                tokenizer.encode(str(prompt) + args.prompt_suffix) for prompt in prompts
            ]
            output0 = model(
                input_ids=torch.tensor(tokenized_prompts[0], device=args.device),
                output_hidden_states=True,
                use_cache=True,
            )

            # Condition 2: Decoder-only transformer
            if is_causal:
                question_len = common_prefix_len(*tokenized_prompts)
                question_hiddens = pytree_map(
                    lambda x: x[:, :question_len], output0.hidden_states
                )

                hiddens = [reduce_seqs(output0.hidden_states)] + [
                    reduce_seqs(
                        model(
                            input_ids=torch.tensor(prompt_ids, device=args.device),
                            output_hidden_states=True,
                            past_key_values=question_hiddens,
                        ).hidden_states
                    )
                    for prompt_ids in tokenized_prompts[1:]
                ]

            # Condition 3: Transformer encoder
            else:
                hiddens = [reduce_seqs(output0.hidden_states)] + [
                    reduce_seqs(
                        model(
                            input_ids=tokenize(str(prompt) + args.prompt_suffix),
                            output_hidden_states=True,
                        ).hidden_states
                    )
                    for prompt in rest
                ]

        # [num_layers, num_choices, hidden_size]
        yield torch.stack(hiddens, dim=1), label
