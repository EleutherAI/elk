from accelerate import find_executable_batch_size
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import cast, Sequence
import pandas as pd
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


@torch.autocast("cuda", enabled=torch.cuda.is_available())
@torch.no_grad()
def extract_hiddens(
    args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataframe: pd.DataFrame,
) -> list[torch.Tensor]:
    """
    Run inference on a model with a set of prompts, collecting the hidden states.

    Returns:
        hidden_states_per_label: a list of two numpy arrays, each of shape (num_data,
        layer, hid_dim), where num_data is the number of data points in the dataframe.
    """

    apply_tokenizer = lambda s: tokenizer(s, return_tensors="pt").input_ids.to(
        args.device
    )
    hidden_states_per_label = [[], []]

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
    for idx in range(len(dataframe)):
        # There are three conditions here:
        # 1) Encoder-decoder transformer, with answer in the decoder
        # 2) Decoder-only transformer
        # 3) Transformer encoder
        # In cases 1 & 2, we can reuse hidden states for the question.
        # In case 3, we have to recompute all hidden states every time.
        ans0, ans1 = dataframe.loc[idx, "selection"]

        # Condition 1: Encoder-decoder transformer, with answer in the decoder
        if is_enc_dec and not args.use_encoder_states:
            # First run the full model on the question + answer 0
            output0 = model(
                input_ids=torch.tensor(dataframe.loc[idx, "0"], device=args.device),
                labels=apply_tokenizer(ans0 + args.prompt_suffix),
                output_hidden_states=True,
            )
            # Then run the decoder on answer 1 with cached encoder states
            output1 = model(
                encoder_hidden_states=output0.encoder_hidden_states,
                labels=apply_tokenizer(ans1),
                output_hidden_states=True,
            )

        # Either a decoder-only transformer or a transformer encoder
        else:
            # First run the model on the question + answer 0
            output0 = model(
                input_ids=apply_tokenizer(dataframe.loc[idx, "0"] + args.prompt_suffix),
                output_hidden_states=True,
                use_cache=True,
            )

            # Condition 2: Decoder-only transformer
            if isinstance(output0, CausalLMOutputWithPast):
                output1 = model(
                    input_ids=apply_tokenizer(ans1 + args.prompt_suffix),
                    output_hidden_states=True,
                    past_key_values=output0.past_key_values,
                )

            # Condition 3: Transformer encoder
            else:
                output1 = model(
                    input_ids=apply_tokenizer(dataframe.loc[idx, "1"]),
                    output_hidden_states=True,
                )

        ids_paired = [
            apply_tokenizer(dataframe.loc[idx, column_name] + args.prompt_suffix)
            for column_name in ["0", "1"]
        ]

        # extract the corresponding token
        for label in range(2):
            # shape (layer * hid_dim)
            res = torch.stack(
                [
                    get_hiddenstate_token(w, args.token_place)
                    for w in hidden_states_paired[label]
                ],
                dim=0,
            )
            hidden_states_per_label[label].append(res)

    # For each list in hidden_states, it's a list with `len(frame)` arrays, and each
    # array has shape `layer * hid_dim`
    # for each list, stack them to `num_data * layer * hid_dim`
    # TODO: WHY ARE WE DOING YET ANOTHER STACKING OPERATION?
    return [torch.stack(w, dim=0) for w in hidden_states_per_label]


@find_executable_batch_size
def _extract_inner(batch_size: int):
    pass


def get_hiddenstate_token(hidden_state, method):
    """
    This function is used to extract the hidden state of a token from the hidden states
    of a sequence given an extraction method.

    Args:
        # TODO: figure out the correct shape for hidden_state
        hidden_state: a tensor of shape (seq_len, batch_size, hid_dim) or (batch_size,
        seq_len, hid_dim)
        method: a string in ["first", "last", "average"]

    Returns:
        a tensor of shape (batch_size, hid_dim) corresponding to the hidden state of
        the token.
    """
    if len(hidden_state.shape) == 3:
        hidden_state = hidden_state[0]
    if method == "first":
        return hidden_state[0]
    elif method == "last":
        return hidden_state[-1]
    elif method == "average":
        return torch.mean(hidden_state, dim=0)
    else:
        raise NotImplementedError(
            "Only support `token_place` in `first`, `last` and `average`!"
        )
