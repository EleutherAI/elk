from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification
)
import os
import torch
import pandas as pd
from utils_generation.construct_prompts import constructPrompt, prompt_dict
from utils_generation.save_utils import get_directory
from datasets import load_dataset
from promptsource.templates import DatasetTemplates



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
        model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{mdl_name}", cache_dir = cache_dir)
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model = GPT2LMHeadModel.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(f"bigscience/{mdl_name}", cache_dir=cache_dir)
    elif "unifiedqa" in mdl_name:
        model = T5ForConditionalGeneration.from_pretrained(f"allenai/{mdl_name}", cache_dir=cache_dir)
    elif "deberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(f"microsoft/{mdl_name}", cache_dir=cache_dir)
    elif "roberta" in mdl_name:
        model = AutoModelForSequenceClassification.from_pretrained(mdl_name, cache_dir = cache_dir)
    elif "t5" in mdl_name:
        model = AutoModelWithLMHead.from_pretrained(mdl_name, cache_dir=cache_dir)
    
    # We only use the models for inference, so we don't need to train them and hence don't need to track gradients
    return model.eval()

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
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
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
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{mdl_name}", cache_dir = cache_dir)
    elif mdl_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = GPT2Tokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "T0" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(f"bigscience/{mdl_name}", cache_dir=cache_dir)
    elif "unifiedqa" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(f"allenai/{mdl_name}", cache_dir=cache_dir)
    elif "deberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{mdl_name}", cache_dir=cache_dir)
    elif "roberta" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)
    elif "t5" in mdl_name:
        tokenizer = AutoTokenizer.from_pretrained(mdl_name, cache_dir=cache_dir)

    return tokenizer

# TODO: UNDEERSTAND THIS FUNCTION
def get_sample_data(dataset_name, split_dataframes, sample_amount):
    '''
    Args:

        dataset_name:   the name of the dataset, some datasets have special token name.
        split_dataframes:  a list of dataframes corresponding to the split of the dataset (train, test, eval etc.)
        num_data:    number of data point we want to sample, default is twice as final size, considering that some examples are too long and could be dropped.
    '''

    label_tag = "label" if dataset_name != "story-cloze" else "answer_right_ending"
    
    all_label_names = set(split_dataframes[0][label_tag].to_list())
    num_labels = len(all_label_names)

    # TODO: UNDEERSTAND THIS FUNCTION
    data_num_lis = get_balanced_sample_num(sample_amount, num_labels)

    # randomize data frac=1 means that we take the return 100% of the dataset
    shuffled_split_dataframes = [dataframe.sample(frac=1).reset_index(drop=True) for dataframe in split_dataframes]

    tmp_lis = []
    prior = shuffled_split_dataframes[0]

    for i, lbl in enumerate(all_label_names):
        # the length of data_list is at most 2
        prior_size = len(prior[prior[label_tag] == lbl])
        if prior_size < data_num_lis[i]:
            tmp_lis.append(pd.concat(
                [prior[prior[label_tag] == lbl], shuffled_split_dataframes[1][shuffled_split_dataframes[1][label_tag] == lbl][: data_num_lis[i] - prior_size]], ignore_index=True))
        else:
            tmp_lis.append(prior[prior[label_tag] == lbl].sample(data_num_lis[i]).reset_index(drop=True))

    return pd.concat(tmp_lis).sample(frac=1).reset_index(drop=True)


def get_balanced_sample_num(sample_amount, num_labels):
    """
    Get the number of samples per label to balance the dataset.
    
    Args:
        sample_amount: int, the number of samples to take.
        num_labels: int, the number of labels in the dataset.
        
    Returns:
        balanced_sample_num: list of int, the number of samples per label.
    """

    samples_per_group = sample_amount // num_labels
    remaining_samples = sample_amount - samples_per_group * num_labels
    balanced_sample_num = []
    for i in range(num_labels):
        if i < num_labels - remaining_samples:
            balanced_sample_num.append(samples_per_group)
        else:
            balanced_sample_num.append(samples_per_group + 1)
    return balanced_sample_num


def get_hugging_face_load_name(dataset_name):
    """
    Get the name of the dataset in the right format for the huggingface datasets library.
    
    Args:
        dataset_name: str, the name of the dataset.
    
    Returns:
        list of str, the name of the dataset in the right format for the huggingface datasets library.
    """
    if dataset_name in ["imdb", "amazon-polarity", "ag-news", "dbpedia-14", "piqa"]:
        return [dataset_name.replace("-", "_")]
    elif dataset_name in ["copa", "rte", "boolq"]:
        return ["super_glue", dataset_name.replace("-", "_")]
    elif dataset_name in ["qnli"]:
        return ["glue", dataset_name.replace("-", "_")]
    elif dataset_name == "story-cloze":
        return ["story_cloze", "2016"]

def get_raw_dataset(dataset_name, cache_dir):
    '''
    This function will load datasets from module or raw csv, and then return a pd DataFrame.
    This DataFrame can be used to construct the example.
    '''
    if dataset_name != "story-cloze":
        raw_dataset = load_dataset(*get_hugging_face_load_name(dataset_name), cache_dir=cache_dir)
    else:
        raw_dataset = load_dataset(*get_hugging_face_load_name(dataset_name),cache_dir=cache_dir, data_dir="./datasets/rawdata")

    if dataset_name in ["imdb", "amazon-polarity", "ag-news", "dbpedia-14"]:
        dataset_split_names = ["test", "train"]
    elif dataset_name in ["copa", "rte", "boolq", "piqa", "qnli"]:
        dataset_split_names = ["validation", "train"]
    elif dataset_name in ["story-cloze"]:
        dataset_split_names = ["test", "validation"]

    return raw_dataset, dataset_split_names


def create_promt_dataframe(dataset, prompt_idx, raw_data, max_num, tokenizer, complete_path, model, prefix, save_base_dir):
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
    prompt_dataframe = constructPrompt(set_name=dataset, frame=raw_data,
                            prompt_idx=prompt_idx, mdl_name=model, 
                            tokenizer=tokenizer, max_num = max_num, 
                            confusion = prefix)

    create_directory(save_base_dir)
    create_directory(complete_path)
    complete_frame_csv_path = os.path.join(complete_path, "frame.csv")
    
    return prompt_dataframe, complete_frame_csv_path

def create_name_to_dataframe(data_base_dir, all_dataset_names, num_prompts_per_dataset, num_data, tokenizer, save_base_dir, model, prefix, token_place, reload_data):
    """
    This function will create a dictionary of dataframes, where the key is the dataset name and the value is the dataframe.

    Args:
        data_base_dir: str, the directory to save the raw data csv files.
        dataset_names: list of str, the names of the datasets.
        num_prompts_per_dataset: list of int, the number of prompts per dataset.
        num_data: int, the number of data points that will be used for each dataset.
        tokenizer: transformers.PreTrainedTokenizer, tokenizer associated with the model.
        save_base_dir: str, the directory to save the prompt dataframes.
        model: str, the name of the model.
        prefix: str, the prefix of the prompt.
        token_place: str, Determine which token's hidden states will be generated. Can be `first` or `last` or `average`.
        reload_data: bool, whether to reload the data.

    Returns:
        name_to_dataframe: dict, the dictionary of dataframes.
    """
    create_directory(data_base_dir)
    name_to_dataframe = {}
    for dataset_name, num_prompts in zip(all_dataset_names, num_prompts_per_dataset):
        for idx in range(num_prompts):
            path = os.path.join(data_base_dir, f"rawdata_{dataset_name}_{num_data}.csv")
        
            dataset_name_with_num = f"{dataset_name}_{num_data}_prompt{idx}"
            complete_path = get_directory(save_base_dir, model, dataset_name_with_num, prefix, token_place)
            dataframe_path = os.path.join(complete_path, "frame.csv")
 
            if os.path.exists(dataframe_path):
                prompt_dataframe = pd.read_csv(dataframe_path, converters={"selection": eval})
                name_to_dataframe[dataset_name_with_num] = prompt_dataframe
            else:  
                cache_dir = os.path.join(data_base_dir, "cache")
                create_directory(cache_dir)

                # This is a dataframe with random order data
                # Can just take enough data from scratch and then stop as needed
                # the length of raw_data will be 2 times as the intended length
                raw_dataset, dataset_split_names = get_raw_dataset(dataset_name, cache_dir)
                split_dataframes = [raw_dataset[split_name].to_pandas() for split_name in dataset_split_names]
                raw_data = get_sample_data(dataset_name, split_dataframes, 2 * num_data)
                raw_data.to_csv(path, index=False)

                prompt_dataframe, complete_frame_csv_path = create_promt_dataframe(dataset_name, idx, raw_data, num_data, tokenizer, 
                                                                   complete_path, model, prefix, save_base_dir) 
                name_to_dataframe[dataset_name_with_num] = prompt_dataframe
                prompt_dataframe.to_csv(complete_frame_csv_path, index = False)

    return name_to_dataframe


def create_directory(name):
    """
    This function will create a directory if it does not exist.
    """
    if not os.path.exists(name):
        os.makedirs(name)

def get_num_templates_per_dataset(dataset_name_list):
    num_templates_per_dataset = []
    for dataset_name in dataset_name_list:
        amount_of_templates = 0
        if dataset_name in prompt_dict.keys():
            amount_of_templates += len(prompt_dict[dataset_name])
        if dataset_name not in ["ag-news", "dbpedia-14"]:
            amount_of_templates += len(DatasetTemplates(*get_hugging_face_load_name(dataset_name)).all_template_names)
        if dataset_name == "copa":
            amount_of_templates -= 4  # do not use the last four prompts
        num_templates_per_dataset.append(amount_of_templates)

    return num_templates_per_dataset