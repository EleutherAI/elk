from numpy import longdouble
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPTNeoForCausalLM,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification
)
import os
import torch
import pandas as pd
from datasets import load_dataset
from utils_generation.construct_prompts import constructPrompt, MyPrompts
from utils_generation.save_utils import saveFrame, getDir


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
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/{}".format(
            mdl_name), cache_dir = cache_dir)
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model = GPT2LMHeadModel.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "bigscience/{}".format(mdl_name), cache_dir=cache_dir)
    elif "unifiedqa" in mdl_name:
        model = T5ForConditionalGeneration.from_pretrained(
            "allenai/" + mdl_name, cache_dir=cache_dir)
    elif "deberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/{}".format(mdl_name), cache_dir=cache_dir)
    elif "roberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(mdl_name, cache_dir = cache_dir)
    elif "t5" in mdl_name:
        model = AutoModelWithLMHead.from_pretrained(mdl_name, cache_dir=cache_dir)
    
    # We only use the models for inference, so we don't need to train them and hence don't need to track gradients
    model.eval()
    
    return model

def put_model_on_device(model, parallelize, device = "cuda"):
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
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        else:
            mps_device = torch.device("mps")
            model.to(mps_device)
    elif parallelize == True:
        model.parallelize()
    elif device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
        else:
            model.to('cuda')
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
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/{}".format(
            mdl_name), cache_dir = cache_dir)
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = GPT2Tokenizer.from_pretrained(
            mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "bigscience/{}".format(mdl_name), cache_dir=cache_dir)
    elif "unifiedqa" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/" + mdl_name, cache_dir=cache_dir)
    elif "deberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/" + mdl_name, cache_dir=cache_dir)
    elif "roberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "t5" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)

    return tokenizer

def get_sample_data(set_name, data_list, total_num):
    '''
        set_name:   the name of the dataset, some datasets have special token name.
        data_list:  a list of dataframe, with order queals to token_list
        max_num:    number of data point that wants to take, default is twice as final size, considering that some examples are too long and could be dropped.
    '''

    lbl_tag = "label" if set_name != "story-cloze" else "answer_right_ending"
    
    label_set = set(data_list[0][lbl_tag].to_list())
    label_num = len(label_set)
    data_num_lis = get_balanced_num(
        total_num=total_num, lis_len=label_num)

    # randomized
    data_list = [w.sample(frac=1).reset_index(drop=True) for w in data_list]

    tmp_lis = []
    prior = data_list[0]

    for i, lbl in enumerate(label_set):
        # the length of data_list is at most 2
        prior_size = len(prior[prior[lbl_tag] == lbl])
        if prior_size < data_num_lis[i]:
            tmp_lis.append(pd.concat(
                [prior[prior[lbl_tag] == lbl], data_list[1][data_list[1][lbl_tag] == lbl][: data_num_lis[i] - prior_size]], ignore_index=True))
        else:
            tmp_lis.append(prior[prior[lbl_tag] == lbl].sample(data_num_lis[i]).reset_index(drop=True))

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
    '''
        This function will load datasets from module or raw csv, and then return a pd DataFrame
        This DataFrame can be used to construct the example
    '''
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
    raw_data = get_sample_data(set_name, [raw_set[w].to_pandas()
                               for w in token_list], 2 * max_num)

    return raw_data


def loadDatasets(args, tokenizer):
    '''
        This fnction will return the datasets, their corresponding name (with prompt suffix, confusion suffix, etc), which should be used to save the hidden states
    '''
    print("-------- datasets --------")
    base_dir = args.data_base_dir
    set_list = args.datasets
    num_data = [int(w) for w in args.num_data]
    confusion = args.prefix
    reload = args.reload_data
    prompt_idx_list = [int(w) for w in args.prompt_idx]

    # deal with the length of prompt_idx_list, and extend
    # end up making prompt_idx_list and set_list with the same length
    if not args.swipe:
        print("Consider datasets {} and prompt idx {}.".format(set_list, prompt_idx_list))
        set_num = len(set_list)
        set_list = [w for w in set_list for _ in range(len(prompt_idx_list))]
        prompt_idx_list = [j for _ in range(set_num) for j in prompt_idx_list]
    else:
        # swipe: for each dataset, will use all the prompts
        prompt_idx_list = MyPrompts.getGlobalPromptsNum(set_list)
        print("Consider datasets {} with {} prompts each.".format(set_list, prompt_idx_list))
        set_list = [[w for _ in range(times)] for w, times in zip(set_list, prompt_idx_list)]
        prompt_idx_list = [[w for w in range(times)] for times in prompt_idx_list]
        set_list, prompt_idx_list = [w for j in set_list for w in j], [w for j in prompt_idx_list for w in j]

    # deal with the length of `num_data`
    # end up making num_data and set_list with the same length
    assert len(num_data) == 1 or len(num_data) == len(
        set_list), "The length of `num_data` should either be one or be the same as `datasets`!"
    if len(num_data) == 1:
        num_data = [num_data[0] for _ in set_list]

    print("Processing {} data points in total.".format(sum(num_data)))

    # create the directory if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    cache_dir = os.path.join(base_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    frame_dict = {}
    reload_set_name = ""    # Only reload if this is the first prompt of a dataset
    for (set_name, prompt_idx, max_num) in zip(set_list, prompt_idx_list, num_data):
        path = os.path.join(
            base_dir, "rawdata_{}_{}.csv".format(set_name, max_num))

                
        # load datasets
        # if complete dataset exists and reload == False, will directly load this dataset
        # Otherwise, load existing raw dataset or reload / load new raw sets
        # notice that this is just the `raw data`, which is a dict or whatever
        dataset_name_w_num = "{}_{}_prompt{}".format(set_name, max_num, prompt_idx)
        complete_path = getDir(dataset_name_w_num, args)
        
        if reload == False and os.path.exists(os.path.join(complete_path, "frame.csv")):
            frame = pd.read_csv(os.path.join(complete_path, "frame.csv"), converters={"selection": eval})
            frame_dict[dataset_name_w_num] = frame
            if args.print_more:
                print("load post-processing {} from {}, length = {}".format(
                    dataset_name_w_num, complete_path, max_num))

        else:   # either reload, or this specific model / confusion args has not been saved yet.
            if (reload == False or reload_set_name == set_name) and os.path.exists(path):
                raw_data = pd.read_csv(path)
                if args.print_more:
                    print("load raw {} from {}, length = {}".format(
                        set_name, path, max_num))
            else:
                if args.print_more:
                    print("load raw dataset {} from module.".format(set_name))
                raw_data = loadFromDatasets(set_name, cache_dir, max_num)
                # save to base_dir, with name `set`+`length`
                # This is only the raw dataset. Saving is just to avoid shuffling every time.
                raw_data.to_csv(path, index=False)
                if args.print_more:
                    print("save raw set to {}".format(path))

            # now start formatting
            # construct the examples according to prompt_ids and so on
            frame = constructPrompt(set_name=set_name, frame=raw_data,
                                    prompt_idx=prompt_idx, mdl_name=args.model, tokenizer=tokenizer, max_num = max_num, confusion = confusion)

            frame_dict[dataset_name_w_num] = frame

            # save this frame
            if not os.path.exists(args.save_base_dir):
                os.mkdir(args.save_base_dir)
            if not os.path.exists(complete_path):
                os.mkdir(complete_path)
            frame.to_csv(os.path.join(complete_path, "frame.csv"), index = False)


        # print an example
        if args.print_more:
            print("[example]:\n{}".format(frame.loc[0, "null"]))


    print("-------- datasets --------")
    return frame_dict
