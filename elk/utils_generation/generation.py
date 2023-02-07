import torch
import numpy as np
import functools


def calculate_hidden_state(args, model, tokenizer, dataframe, mdl_name):
    """
    This function is used to calculate the hidden states of a sequence given a model tokenizer and a dataframe.

    Args:
        args: a Namespace object containing the arguments.
        model: a huggingface model.
        tokenizer: a huggingface tokenizer.
        dataframe: a pandas dataframe containing the data.
        mdl_name: a string containing the name of the model.

    Returns:
        hidden_states_per_label: a list of two numpy arrays, each of shape (num_data, layer, hid_dim), where num_data is the number of data points in the dataframe.
    """

    if (
        args.states_location == "decoder"
    ):  # In such a case, the program should generate decoder hidden states
        assert (
            "T0" in mdl_name or "t5" in mdl_name or "gpt" in mdl_name
        ), NotImplementedError(
            f"BERT does not have decoder. Relevant args: model={mdl_name}, states_location={args.states_location}."
        )

    if args.states_location == "encoder":
        assert "gpt" not in mdl_name, NotImplementedError(
            f"GPT model does not have encoder. Relevant args: model={mdl_name}, states_location={args.states_location}."
        )

    apply_tokenizer = functools.partial(
        tokenize_to_gpu, tokenizer=tokenizer, device=args.model_device
    )
    pad_answer = apply_tokenizer("")

    hidden_states_per_label = [[], []]
    for idx in range(len(dataframe)):
        # calculate the hidden states
        if "T0" in mdl_name or "unifiedqa" in mdl_name or "t5" in mdl_name:
            if args.states_location == "encoder":
                ids_paired = [
                    apply_tokenizer(get_datapoint_from_df(dataframe, idx, column_name))
                    for column_name in ["0", "1"]
                ]
                hidden_states_paired = [
                    model(
                        ids, labels=pad_answer, output_hidden_states=True
                    ).encoder_hidden_states
                    for ids in ids_paired
                ]
            else:
                ans_token = [
                    apply_tokenizer(w)
                    for w in get_datapoint_from_df(dataframe, idx, "selection")
                ]
                input_ids = apply_tokenizer(
                    get_datapoint_from_df(dataframe, idx, "null")
                )
                hidden_states_paired = [
                    model(
                        input_ids, labels=ans, output_hidden_states=True
                    ).decoder_hidden_states
                    for ans in ans_token
                ]
        elif "gpt" in mdl_name or "bert" in mdl_name:
            appender = " " + str(tokenizer.eos_token) if "gpt" in mdl_name else ""
            ids_paired = [
                apply_tokenizer(
                    get_datapoint_from_df(dataframe, idx, column_name) + appender
                )
                for column_name in ["0", "1"]
            ]
            # Notice that since gpt and bert only have either decoder or encoder, we don't need to specify which one to use.
            hidden_states_paired = [
                model(ids, output_hidden_states=True).hidden_states
                for ids in ids_paired
            ]
        else:
            raise NotImplementedError(f"model {mdl_name} is not supported!")

        # extract the corresponding token
        for label in range(2):
            # shape (layer * hid_dim)
            res = np.stack(
                [
                    torch_to_cpu_np(get_hiddenstate_token(w, args.token_place))
                    for w in hidden_states_paired[label]
                ],
                axis=0,
            )
            hidden_states_per_label[label].append(res)

    # For each list in hidden_states, it's a list with `len(frame)` arrays, and each array has shape `layer * hid_dim`
    # for each list, stack them to `num_data * layer * hid_dim`
    # TODO: WHY ARE WE DOING YET ANOTHER STACKING OPERATION?
    hidden_states_per_label = [np.stack(w, axis=0) for w in hidden_states_per_label]

    return hidden_states_per_label


def get_datapoint_from_df(dataframe, idx, column_name):
    if type(idx) == list:
        return dataframe.loc[idx[0] : idx[1] - 1, column_name]
    return dataframe.loc[idx, column_name]


def get_hiddenstate_token(hidden_state, method):
    """
    This function is used to extract the hidden state of a token from the hidden states of a sequence given an extraction method.

    Args:
        # TODO: figure out the correct shape for hidden_state
        hidden_state: a tensor of shape (seq_len, batch_size, hid_dim) or (batch_size, seq_len, hid_dim)
        method: a string in ["first", "last", "average"]

    Returns:
        a tensor of shape (batch_size, hid_dim) corresponding to the hidden state of the token.
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


def tokenize_to_gpu(s, tokenizer, device):
    return tokenizer(s, return_tensors="pt").input_ids.to(device)


def torch_to_cpu_np(tensor):
    """
    Puts a tensor or a list of tensors on a cpu and converts them into a numpy array.
    """
    if type(tensor) == list:
        return [t.cpu().numpy() for t in tensor]
    return tensor.cpu().numpy()
