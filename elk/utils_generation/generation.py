from .save_utils import save_array, save_records
import functools
import numpy as np
import time
import torch


def cal_zero_and_hidden_states(model, tokenizer, frame_dict, args):
    """
    Calculate the zero-shot accuracy for each dataset and save it as a CSV
    """

    print("-------- zero-shot & generation --------")
    # create records, will save as csv in the end
    records = [
        {
            "model": args.model,
            "dataset": key,
            "prefix": args.prefix,
            "tag": args.tag,
            "cal_zeroshot": bool(args.cal_zeroshot),
            "cal_hiddenstates": bool(args.cal_hiddenstates),
        }
        for key in frame_dict.keys()
    ]

    mdl_name = args.model
    tokenize = functools.partial(get_token, tokenizer=tokenizer)
    with torch.no_grad():
        pad_answer = tokenize("")
        for key, record in zip(frame_dict.keys(), records):
            start = time.time()
            frame = frame_dict[key]

            record["population"] = len(frame)
            if args.print_more:
                print("Start {}, length = {}".format(key, len(frame)))

            # This part corresponds to zero-shot accuracy calculation
            # as well as the logits calculation
            if args.cal_zeroshot or args.cal_logits:
                log_probs_list = []
                logits_list = []

                # Loop over data points
                for idx in range(len(frame)):
                    if "gpt" in mdl_name or "bert" in mdl_name:
                        ans_list = get_data_point(frame, idx, "selection")
                        ans_token = [tokenize(w) for w in ans_list]

                        # get the input_ids
                        ids_paired = [
                            tokenize(get_data_point(frame, idx, w)) for w in ["0", "1"]
                        ]
                        logits = [model(w)["logits"] for w in ids_paired]

                        logit_token = ids_paired

                    elif (
                        "T0" in mdl_name or "unifiedqa" in mdl_name or "t5" in mdl_name
                    ):
                        # for idx in range(len(frame)):

                        ans_list = get_data_point(frame, idx, "selection")
                        ans_token = [tokenize(w) for w in ans_list]
                        # get the input_ids for candidates
                        input_ids = tokenize(get_data_point(frame, idx, "null"))

                        # calculate the logits
                        logits = [
                            model(input_ids, labels=w)["logits"] for w in ans_token
                        ]

                        logit_token = ans_token

                    else:
                        raise NotImplementedError(
                            "model {} is not supported!".format(mdl_name)
                        )

                    if args.cal_zeroshot:
                        # calculate the logits
                        probs = get_log_probs(logits, ans_token, mdl_name)
                        log_probs_list.append(to_np(probs))
                    if args.cal_logits:
                        logits_list.append(
                            to_np(get_logits(logits, logit_token, mdl_name))
                        )

                # Essemble and save
                if args.cal_zeroshot:
                    # add to the records
                    labels = get_data_point(frame, [0, len(frame)], "label").to_list()
                    record["log_probs"] = sum(
                        [int(w[0] < w[1]) == l for w, l in zip(log_probs_list, labels)]
                    ) / len(frame)
                    record["calibrated"] = get_calibrated(log_probs_list, labels)

                    if args.print_more:
                        print(
                            "Finish calculating the zero-shot accuracy for {} data"
                            " in {}.".format(len(frame), key)
                        )

                # save logits
                if args.cal_logits:
                    # notice that logits should be num_data * vocab_size!!!
                    logits = np.stack(logits_list, axis=0)

                    save_array([logits], ["logits"], key, args)

                    if args.print_more:
                        print(
                            "Finish generating logits for {} data in {}. Shape of"
                            " logits is {}.".format(len(frame), key, logits.shape)
                        )

            # This part corresponds to hidden states generation
            if args.cal_hiddenstates:
                hidden_states = [[], []]
                if args.print_more:
                    print(
                        "Generating {} hidden states for {}. Layer = {}".format(
                            args.states_location, mdl_name, args.states_index
                        )
                    )
                if (
                    args.states_location == "decoder"
                ):  # In suce case, the program should generate decoder hidden states
                    assert (
                        "T0" in mdl_name or "t5" in mdl_name or "gpt" in mdl_name
                    ), NotImplementedError(
                        "BERT does not have decoder. Relevant args: model={},"
                        " states_location={}.".format(mdl_name, args.states_location)
                    )
                if args.states_location == "encoder":
                    assert "gpt" not in mdl_name, NotImplementedError(
                        "GPT model does not have encoder. Relevant args: model={},"
                        " states_location={}.".format(mdl_name, args.states_location)
                    )
                for idx in range(len(frame)):
                    # calculate the hidden states
                    if "T0" in mdl_name or "unifiedqa" in mdl_name or "t5" in mdl_name:
                        if args.states_location == "encoder":
                            ids_paired = [
                                tokenize(get_data_point(frame, idx, w))
                                for w in ["0", "1"]
                            ]
                            hidden_states_paired = [
                                model(
                                    ids, labels=pad_answer, output_hidden_states=True
                                ).encoder_hidden_states
                                for ids in ids_paired
                            ]
                        else:
                            ans_token = [
                                tokenize(w)
                                for w in get_data_point(frame, idx, "selection")
                            ]
                            # get the input_ids for candidates
                            input_ids = tokenize(get_data_point(frame, idx, "null"))

                            # calculate the hidden states and take the layer `state_idx`
                            hidden_states_paired = [
                                model(
                                    input_ids, labels=answer, output_hidden_states=True
                                ).decoder_hidden_states
                                for answer in ans_token
                            ]
                    elif "gpt" in mdl_name or "bert" in mdl_name:
                        appender = (
                            " " + str(tokenizer.eos_token) if "gpt" in mdl_name else ""
                        )
                        ids_paired = [
                            tokenize(get_data_point(frame, idx, w) + appender)
                            for w in ["0", "1"]
                        ]
                        # Notice that since gpt and bert only have either decoder or
                        # encoder, we don't need to specify which one to use.
                        hidden_states_paired = [
                            model(ids, output_hidden_states=True).hidden_states
                            for ids in ids_paired
                        ]
                    else:
                        raise NotImplementedError(
                            "model {} is not supported!".format(mdl_name)
                        )

                    # extract the corresponding token
                    for i in range(2):
                        # shape (layer * hid_dim)
                        hidden_states[i].append(
                            np.stack(
                                [
                                    to_np(get_states_token(w, args.token_place))
                                    for w in hidden_states_paired[i]
                                ],
                                axis=0,
                            )
                        )

                # For each list in hidden_states, it's a list with `len(frame)` arrays,
                # and each array has shape `layer * hid_dim`
                # for each list, stack them to `num_data * layer * hid_dim`
                hidden_states = [np.stack(w, axis=0) for w in hidden_states]

                save_array(hidden_states, ["0", "1"], key, args)

                if args.print_more:
                    print(
                        "Finish generating hidden states for {} data in {}.".format(
                            len(frame), key
                        )
                    )

            print(
                "{} s/data. {}".format(
                    round((time.time() - start) / record["population"], 2), record
                )
            )

    # save records
    save_records(records, args)
    print("-------- zero-shot & generation --------")


def get_data_point(frame, idx, key):
    if type(idx) == list:
        return frame.loc[idx[0] : idx[1] - 1, key]
    return frame.loc[idx, key]


def get_states_token(hidden_state, method):
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


def get_token(s, tokenizer):
    return tokenizer(s, return_tensors="pt").input_ids.to("cuda")


def get_log_probs(logits, ans_token, mdl_name):
    """
    `logits` is a list of logit generated by the model. Each element has shape
        `1 * #token * vocab_size` or `#token * vocab_size`.
    `ans_token` is also a list with the same length as logits. Each element
        corresponds to the token of answer.
    `mdl_name` is the name of the model. The behavior of function will change
        according to this.
        if `mdl_name` == unifiedqa or T0 or T5, then last token is eos, should ignore
        if `mdl_name` == gpt, then does not consider the last token either.
        if `mdl_name` == roberta or deberta, then logits are just NLI
            ([0] is contradiction and [-1] is entailment. Directly return [-1] - [0])

    Return a list, and each element corresponds to the log_probs of answering with
    this answer.

    Eaxmple:
        # Assume we use T0pp, then when generating logits we need to provide labels,
        # as shown below.
        mdl_name = "T0pp"
        ans_list = ["negative", "positive"]
        ans_token = [get_token(w, tokenizer) for w in ans_list]
        logits = [model(question, labels = w)["logits"] for w in ans_token]

        logprobs = get_log_probs(logits, ans_token, mdl_name)
    """

    logits = [w[0] for w in logits] if len(logits[0].shape) == 3 else logits
    if "gpt" in mdl_name:
        return [
            torch.mean(
                logit.log_softmax(dim=-1)[range(-answer.shape[1] - 1, -1), answer[0]]
            )
            for logit, answer in zip(logits, ans_token)
        ]
    elif "bert" in mdl_name:
        # each element has shape 1 * 3, and doing softmax will return probs
        # TODO: This was in the original code but `tmp` is never read. Is this a bug?
        # tmp = [w.softmax(-1) for w in logits]

        # entailment - contradiction, representing how this is likely to be the answer
        return [w[0, -1] - w[0, 0] for w in logits]
    else:
        return [
            torch.mean(
                logit.log_softmax(dim=-1)[range(answer.shape[1] - 1), answer[0][:-1]]
            )
            for logit, answer in zip(logits, ans_token)
        ]


def get_first_diff(x, y):
    """
    get the location of the first difference between the tokens x and y.
    """
    minlen = min(len(x), len(y))
    for i in range(minlen):
        if x[i] != y[i]:
            return i
    return minlen


def get_logits(logits, token_lis, mdl_name):
    """
    Takes the logits output by the model, the list of tokens and the model name.
    `logits` is a list, and each element is a tensor with shape
        `1 * #tokens * vocav_size`.
    `token_lis` is also a list with the same length, and each element is a
        `1 * #token` tensor
    `mdl_name` is the name of the model. The behavior of function will change according
        to this.

    Return the correct logits sliece, i.e. the logits of token that is going to make the
    first different prediction between two answers.

    Example:
        mdl_name = "gpt-j-6B" # assume we use gpt type model here
        ans_list = ["negative", "positive"]
        # get the input_ids, here `get_data_point(frame, idx, w)` can be "imagined" as
        # text = question + ans_list[w]
        ids_paired = [
            get_token(get_data_point(frame, idx, w), tokenizer)
            for w in ["0", "1"]
        ]
        logits = [model(w)["logits"] for w in ids_paired]

        logits_slice = get_logits(logits, ids_paired, mdl_name)

    """
    diff = get_first_diff(token_lis[0][0], token_lis[1][0])
    diff = diff - 1 if "gpt" in mdl_name else diff
    return logits[0][0, diff, :]


def to_np(x):
    if type(x) == list:
        return [w.cpu().numpy() for w in x]
    return x.cpu().numpy()


def get_calibrated(log_probs_list, labels):
    """
    This function takes the list of log_probs and labels, and give the calibrated
    zero-shot accuracy.
    `log_prob_list` is a list, and each element is a binary tuple, with each value
        corresponding to the log_probs of that labels
    `labels` is a list, and each element is the correct label. This is just used to
        calculate the accuracy.
    """
    log_delta = [w[0] - w[1] for w in log_probs_list]
    median = np.median(log_delta)
    correctness = sum([int(w < median) == u for w, u in zip(log_delta, labels)])
    return correctness / len(log_delta)
