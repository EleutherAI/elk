from numpy import longdouble
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
)
import os
import torch
import pandas as pd
from datasets import load_dataset
from .construct_prompts import constructPrompt, MyPrompts
from .save_utils import saveFrame, getDir
from .save_utils import save_records_to_csv
from pathlib import Path


def load_model(mdl_name, cache_dir):
    """
    Load model from cache_dir or from HuggingFace model hub.

    Args:
        mdl_name (str): name of the model
        cache_dir (str): path to the cache directory

    Returns:
        model (torch.nn.Module): model
    """
    if mdl_name in ["gpt-neo-2.7B", "gpt-j-6B"]:
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/{}".format(mdl_name), cache_dir=cache_dir
        )
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model = GPT2LMHeadModel.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "bigscience/{}".format(mdl_name), cache_dir=cache_dir
        )
    elif "unifiedqa" in mdl_name:
        model = T5ForConditionalGeneration.from_pretrained(
            "allenai/" + mdl_name, cache_dir=cache_dir
        )
    elif "deberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/{}".format(mdl_name), cache_dir=cache_dir
        )
    elif "roberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            mdl_name, cache_dir=cache_dir
        )
    elif "t5" in mdl_name:
        model = AutoModelWithLMHead.from_pretrained(mdl_name, cache_dir=cache_dir)
    else:
        # TODO add a string of try/excepts, figure out a better model name format later
        model = AutoModelForCausalLM.from_pretrained(mdl_name, cache_dir=cache_dir)

    # We only use the models for inference,
    # so we don't need to train them and hence don't need to track gradients
    model.eval()

    return model


def put_model_on_device(model, parallelize, device="cuda"):
    """
    Put model on device.

    Args:
        model (torch.nn.Module): model to put on device
        parallelize (bool): whether to parallelize the model
        device (str): device to put the model on

    Returns:
        model (torch.nn.Module): model on device
    """
    if device == "mps":
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            mps_device = torch.device("mps")
            model.to(mps_device)
    elif parallelize:
        model.parallelize()
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
        else:
            model.to("cuda")
    else:
        model.to("cpu")

    return model


def load_tokenizer(mdl_name, cache_dir):
    """
    Load tokenizer for the model.

    Args:
        mdl_name (str): name of the model
        cache_dir (str): path to the cache directory

    Returns:
        tokenizer: tokenizer for the model
    """

    if mdl_name in ["gpt-neo-2.7B", "gpt-j-6B"]:
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/{}".format(mdl_name), cache_dir=cache_dir
        )
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = GPT2Tokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "bigscience/{}".format(mdl_name), cache_dir=cache_dir
        )
    elif "unifiedqa" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/" + mdl_name, cache_dir=cache_dir
        )
    elif "deberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/" + mdl_name, cache_dir=cache_dir
        )
    # TODO these lines are unnecessary
    elif "roberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "t5" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)

    return tokenizer


def get_sample_data(set_name, data_list, total_num):
    """

    set_name:   the name of the dataset, some datasets have special token name.
    data_list:  a list of dataframe, with order queals to token_list
    max_num:    number of data point that wants to take,
    default is twice as final size, considering that
    some examples are too long and could be dropped.
    """

    lbl_tag = "label" if set_name != "story-cloze" else "answer_right_ending"

    label_set = set(data_list[0][lbl_tag].to_list())
    label_num = len(label_set)
    data_num_lis = get_balanced_num(total_num=total_num, lis_len=label_num)

    # randomized
    data_list = [w.sample(frac=1).reset_index(drop=True) for w in data_list]

    tmp_lis = []
    prior = data_list[0]

    for i, lbl in enumerate(label_set):
        # the length of data_list is at most 2
        prior_size = len(prior[prior[lbl_tag] == lbl])
        if prior_size < data_num_lis[i]:
            tmp_lis.append(
                pd.concat(
                    [
                        prior[prior[lbl_tag] == lbl],
                        data_list[1][data_list[1][lbl_tag] == lbl][
                            : data_num_lis[i] - prior_size
                        ],
                    ],
                    ignore_index=True,
                )
            )
        else:
            tmp_lis.append(
                prior[prior[lbl_tag] == lbl]
                .sample(data_num_lis[i])
                .reset_index(drop=True)
            )

    return pd.concat(tmp_lis).sample(frac=1).reset_index(drop=True)


def get_balanced_num(total_num, lis_len):
    tmp = total_num // lis_len
    more = total_num - tmp * lis_len
    return [tmp if i < lis_len - more else tmp + 1 for i in range(lis_len)]


def getLoadName(set_name):
    if set_name in ["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "piqa"]:
        return [set_name.replace("-", "_")]
    elif set_name in ["copa", "rte", "boolq"]:
        return ["super_glue", set_name.replace("-", "_")]
    elif set_name in ["qnli"]:
        return ["glue", set_name.replace("-", "_")]
    elif set_name == "story-cloze":
        return ["story_cloze", "2016"]


def loadFromDatasets(set_name, cache_dir, max_num):
    """
    This function will load datasets from module or raw csv,
    and then return a pd DataFrame
    This DataFrame can be used to construct the example
    """
    if set_name != "story-cloze":
        raw_set = load_dataset(*getLoadName(set_name))
    else:
        raw_set = load_dataset(*getLoadName(set_name), data_dir="./datasets/rawdata")

    if set_name in ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]:
        token_list = ["test", "train"]
    elif set_name in ["copa", "rte", "boolq", "piqa", "qnli"]:
        token_list = ["validation", "train"]
    elif set_name in ["story-cloze"]:
        token_list = ["test", "validation"]

    # This is a dataframe with random order data
    # Can just take enough data from scratch and then stop as needed
    # the length of raw_data will be 2 times as the intended length
    raw_data = get_sample_data(
        set_name, [raw_set[w].to_pandas() for w in token_list], 2 * max_num
    )

    return raw_data


def setup_dataset_names_and_prompt_idx(prompt_idxs=None, dataset_names=None):
    """
    This function will setup the dataset_names and prompt_idxs.
    If swipe is True, then will use all the prompts for each dataset.
    If swipe is False, then will use the prompt_idxs for each dataset.

    Args:
        swipe: bool, if True, will use all the prompts for each dataset.
        prompt_idxs: list of int, the prompt idxs that will be used.
        dataset_names: list of str, the dataset names that will be used.

    Returns:
        dataset_names: list of str, the dataset names that will be used.
        prompt_idxs: list of int, the prompt idxs that will be used.
    """

    prompt_idxs = MyPrompts.getGlobalPromptsNum(dataset_names)

    print(
        "Consider datasets {} with {} prompts each.".format(dataset_names, prompt_idxs)
    )
    dataset_names = [
        [w for _ in range(times)] for w, times in zip(dataset_names, prompt_idxs)
    ]
    prompt_idxs = [[w for w in range(times)] for times in prompt_idxs]
    dataset_names, prompt_idxs = [w for j in dataset_names for w in j], [
        w for j in prompt_idxs for w in j
    ]

    return dataset_names, prompt_idxs


def create_directory(name):
    """
    This function will create a directory if it does not exist.

    Args:
        name: str, the name of the directory.

    Returns:
        None
    """
    name.mkdir(parents=True, exist_ok=True)


def create_and_save_promt_dataframe(
    args, dataset, prompt_idx, raw_data, max_num, tokenizer, complete_path
):
    """
    This function will create a prompt dataframe and saves it.

    Args:
        args: argparse, the arguments.
        dataset: str, the name of the dataset.
        prompt_idx: int, the prompt idx.
        raw_data: pd.DataFrame, the raw data.
        max_num: int, the number of data points that will be used for each dataset.
        tokenizer: transformers.PreTrainedTokenizer, the tokenizer.
        complete_path: str, the path to save the prompt dataframe.

    Returns:
        prompt_dataframe: pd.DataFrame, the prompt dataframe.
    """
    prompt_dataframe = constructPrompt(
        set_name=dataset,
        frame=raw_data,
        prompt_idx=prompt_idx,
        mdl_name=args.model,
        tokenizer=tokenizer,
        max_num=max_num,
        confusion=args.prefix,
    )

    create_directory(args.save_base_dir)
    create_directory(complete_path)
    complete_frame_csv_path = complete_path / "frame.csv"
    prompt_dataframe.to_csv(complete_frame_csv_path, index=False)

    return prompt_dataframe


def create_dataframe_dict(
    args,
    data_base_dir,
    dataset_names,
    prompt_idxs,
    num_data,
    tokenizer,
    print_more=False,
):
    """
    This function will create a dictionary of dataframes,
    where the key is the dataset name and the value is the dataframe.

    Args:
        args: argparse, the arguments.
        data_base_dir: str, the directory of the data.
        dataset_names: list of str, the dataset names that will be used.
        prompt_idxs: list of int, the prompt idxs that will be used.
        num_data: list of int, the number of
        data points that will be used for each dataset.
        tokenizer: transformers.PreTrainedTokenizer, the tokenizer.
        print_more: bool, if True, will print more information.

    Returns:
        name_to_dataframe: dict, the dictionary of dataframes.
    """
    create_directory(data_base_dir)
    name_to_dataframe = {}
    reload_set_name = ""  # Only reload if this is the first prompt of a dataset
    for (dataset, prompt_idx, max_num) in zip(dataset_names, prompt_idxs, num_data):
        path = data_base_dir / f"rawdata_{dataset}_{max_num}.csv"

        # load datasets
        # if complete dataset exists and reload == False,
        # will directly load this dataset
        # Otherwise, load existing raw dataset or
        # reload / load new raw sets
        # notice that this is just the
        # `raw data`, which is a dict or whatever
        dataset_name_with_num = f"{dataset}_{max_num}_prompt{prompt_idx}"
        complete_path = getDir(dataset_name_with_num, args)
        dataframe_path = complete_path / "frame.csv"

        if args.reload_data is False and dataframe_path.exists():
            frame = pd.read_csv(dataframe_path, converters={"selection": eval})
            name_to_dataframe[dataset_name_with_num] = frame
            if print_more:
                print(
                    f"load post-processing {dataset_name_with_num} from"
                    f" {complete_path}, length = {max_num}"
                )
        # either reload, or this specific model / confusion args has not been saved yet.
        else:
            if (
                args.reload_data is False or reload_set_name == dataset
            ) and path.exists():
                raw_data = pd.read_csv(path)
                if print_more:
                    print(f"load raw {dataset} from {path}, length = {max_num}")
            else:
                if print_more:
                    print(f"load raw dataset {dataset} from module.")

                cache_dir = data_base_dir / "cache"
                create_directory(cache_dir)
                raw_data = loadFromDatasets(dataset, cache_dir, max_num)
                raw_data.to_csv(path, index=False)

                if print_more:
                    print(f"save raw set to {path}")

            prompt_dataframe = create_and_save_promt_dataframe(
                args, dataset, prompt_idx, raw_data, max_num, tokenizer, complete_path
            )

            name_to_dataframe[dataset_name_with_num] = prompt_dataframe

    return name_to_dataframe


def align_datapoints_amount(num_examples, dataset_names):
    """
    This function will check the length of `num_examples`
    and make it the same as `dataset_names`.
    This function gives us a list of
    num_examples for each prompt template e.g.
    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    ... so, for 13 prompts we obtain 1000 datapoints
    Args:
        num_examples: list of ints,
        the number of data points that will be used for each dataset.
        dataset_names: list of strings,
        the dataset names that will be used.

    Returns:
        num_examples: list of int,
        the number of data points that will be used for each dataset.
    """
    # deal with the length of `num_examples`
    # end up making num_examples and set_list with the same length
    assert len(num_examples) == 1 or len(num_examples) == len(
        dataset_names
    ), "The length of `num_examples` should either be one or be the same as `datasets`!"

    if len(num_examples) == 1:
        num_examples = [num_examples[0] for _ in dataset_names]

    print(f"Processing {num_examples} data points in total.")
    return num_examples


def load_datasets(args, tokenizer):
    """
    This function will return the datasets.
    Their corresponding name will include the prompt suffix, confusion suffix, etc.
    These should be used to save the hidden states.

    Args:
        args: argparse, the arguments.
        tokenizer: transformers.PreTrainedTokenizer, the tokenizer.

    Returns:
        frame_dict: dict, the dictionary of dataframes.
    """
    data_base_dir = args.data_base_dir
    dataset_names = args.datasets
    num_data = [int(w) for w in args.num_data]
    prompt_idxs = [int(w) for w in args.prompt_idx]

    dataset_names, prompt_idxs = setup_dataset_names_and_prompt_idx(
        prompt_idxs, dataset_names
    )

    num_data = align_datapoints_amount(num_data, dataset_names)

    frame_dict = create_dataframe_dict(
        args,
        data_base_dir,
        dataset_names,
        prompt_idxs,
        num_data,
        tokenizer,
        print_more=True,
    )

    return frame_dict
