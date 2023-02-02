import torch
import numpy as np
import functools
from tqdm import tqdm
from .save_utils import saveArray
from .save_utils import save_records_to_csv


def calculate_hidden_state(args, model, tokenizer, frame, mdl_name):

    apply_tokenizer = functools.partial(
        getToken, tokenizer=tokenizer, device=args.model_device
    )

    hidden_states = [[], []]
    pad_answer = apply_tokenizer("")

    if (
        args.states_location == "decoder"
    ):  # In suce case, the program should generate decoder hidden states
        assert (
            "T0" in mdl_name or "t5" in mdl_name or "gpt" in mdl_name
        ), NotImplementedError(
            f"BERT does not have decoder. Relevant args: model={mdl_name},"
            f" states_location={args.states_location}."
        )

    if args.states_location == "encoder":
        assert "gpt" not in mdl_name, NotImplementedError(
            "GPT model does not have encoder. Relevant args: model={mdl_name},"
            " states_location={args.states_location}."
        )

    for idx in range(len(frame)):
        # calculate the hidden states
        if "T0" in mdl_name or "unifiedqa" in mdl_name or "t5" in mdl_name:
            if args.states_location == "encoder":
                ids_paired = [
                    apply_tokenizer(getDataPoint(frame, idx, w)) for w in ["0", "1"]
                ]
                hidden_states_paired = [
                    model(
                        ids, labels=pad_answer, output_hidden_states=True
                    ).encoder_hidden_states
                    for ids in ids_paired
                ]
            else:
                answer_token = [
                    apply_tokenizer(w) for w in getDataPoint(frame, idx, "selection")
                ]
                # get the input_ids for candidates
                input_ids = apply_tokenizer(getDataPoint(frame, idx, "null"))

                # calculate the hidden states and take the layer `state_idx`
                hidden_states_paired = [
                    model(
                        input_ids, labels=a, output_hidden_states=True
                    ).decoder_hidden_states
                    for a in answer_token
                ]
        elif "gpt" in mdl_name or "bert" in mdl_name:
            appender = " " + str(tokenizer.eos_token) if "gpt" in mdl_name else ""
            ids_paired = [
                apply_tokenizer(getDataPoint(frame, idx, w) + appender)
                for w in ["0", "1"]
            ]
            # Notice that since gpt and bert only have either decoder or encoder,
            # we don't need to specify which one to use.
            hidden_states_paired = [
                model(ids, output_hidden_states=True).hidden_states
                for ids in ids_paired
            ]
        else:
            raise NotImplementedError(f"model {mdl_name} is not supported!")

        # extract the corresponding token
        for i in range(2):
            # shape (layer * hid_dim)
            hidden_states[i].append(
                np.stack(
                    [
                        toNP(getStatesToken(w, args.token_place))
                        for w in hidden_states_paired[i]
                    ],
                    axis=0,
                )
            )

    # For each list in hidden_states, it's a list with `len(frame)` arrays,
    # and each array has shape `layer * hid_dim`
    # for each list, stack them to `num_data * layer * hid_dim`
    hidden_states = [np.stack(w, axis=0) for w in hidden_states]

    return hidden_states


def create_hiddenstates(model, tokenizer, name_to_dataframe, args):
    """
    This function will calculate the zeroshot
    accuracy for each dataset and properly store
    """
    with torch.no_grad():
        for name, dataframe in name_to_dataframe.items():
            # This part corresponds to hidden states generation
            hidden_states = calculate_hidden_state(
                args, model, tokenizer, dataframe, args.model
            )
            saveArray(hidden_states, ["0", "1"], name, args)


def create_records(model, tokenizer, name_to_dataframe, args):
    """
    This function will calculate the zeroshot accuracy
    for each dataset and properly store
    """

    # create records, will save as csv in the end
    records = [
        {
            "model": args.model,
            "dataset": key,
            "prefix": args.prefix,
            "tag": args.tag,
            "cal_hiddenstates": bool(args.cal_hiddenstates),
        }
        for key in name_to_dataframe.keys()
    ]

    with torch.no_grad():
        for name, record in tqdm(
            zip(name_to_dataframe.keys(), records),
            desc="Iterating over datasets:",
            position=1,
            leave=False,
        ):
            dataframe = name_to_dataframe[name]
            record["population"] = len(dataframe)

    save_records_to_csv(records, args)


def getDataPoint(frame, idx, key):
    if type(idx) == list:
        return frame.loc[idx[0] : idx[1] - 1, key]
    return frame.loc[idx, key]


def getStatesToken(hidden_state, method):
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


def getToken(s, tokenizer, device):
    return tokenizer(s, return_tensors="pt").input_ids.to(device)


def toNP(x):
    if type(x) == list:
        return [w.cpu().numpy() for w in x]
    return x.cpu().numpy()
