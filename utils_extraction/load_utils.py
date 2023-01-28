import os
import numpy as np
import pandas as pd
import json

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_models = global_dict["registered_models"]
registered_prefix = global_dict["registered_prefix"]
models_layer_num = global_dict["models_layer_num"]


def getDirList(mdl, set_name, load_dir, num_data, confusion, place, prompt_idx):
    length = len(mdl)
    filter = [w for w in os.listdir(load_dir) if (mdl == w[:length] and mdl + "_" in w and set_name + "_" in w and str(num_data) + "_" in w and confusion + "_" in w and place in w)]
    if prompt_idx is not None:     
        filter = [w for w in filter if int(w.split("_")[3][6:]) in prompt_idx]
    return [os.path.join(load_dir, w) for w in filter]

def organizeStates(lis, mode):
    '''
        Whether to do minus, to concat or do nothing
    '''
    if mode in ["0", "1"]:
        return lis[int(mode)]
    elif mode == "minus":
        return lis[0] - lis[1]
    elif mode == "concat":
        return np.concatenate(lis, axis = -1)
    else:
        raise NotImplementedError("This mode is not supported.")

def normalize(data, scale =True, demean = True):
    # demean the array and rescale each data point
    data = data - np.mean(data, axis = 0) if demean else data
    if not scale:
        return data
    norm = np.linalg.norm(data, axis=1)
    avgnorm = np.mean(norm)
    return data / avgnorm * np.sqrt(data.shape[1])

def loadHiddenStates(mdl, set_name, load_dir, promtpt_idx, location = "encoder", layer = -1, num_data = 1000, confusion = "normal", place = "last", scale = True, demean = True, mode = "minus", verbose = True):
    '''
        Load generated hidden states, return a dict where key is the dataset name and values is a list. Each tuple in the list is the (x,y) pair of one prompt.
        if mode == minus, then get h - h'
        if mode == concat, then get np.concatenate([h,h'])
        elif mode == 0 or 1, then get h or h'
    '''

    dir_list = getDirList(mdl, set_name, load_dir, num_data, confusion, place, promtpt_idx)
    append_list = ["_" + location + str(layer) for _ in dir_list]
    
    hidden_states = [
                        organizeStates(
                            [np.load(os.path.join(w, "0{}.npy".format(app))), 
                            np.load(os.path.join(w, "1{}.npy".format(app)))], 
                            mode = mode) 
                        for w, app in zip(dir_list, append_list)
                    ]

    # normalize
    hidden_states = [normalize(w, scale, demean) for w in hidden_states]
    if verbose:        
        print("{} prompts for {}, with shape {}".format(len(hidden_states), set_name, hidden_states[0].shape))
    labels = [np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list()) for w in dir_list]

    return [(u,v) for u,v in zip(hidden_states, labels)]


#TODO: UNDESTSTAND THIS 
def getPermutation(data_list, rate = 0.6):
    length = len(data_list[0][1])
    permutation = np.random.permutation(range(length)).reshape(-1)
    return [permutation[: int(length * rate)], permutation[int(length * rate):]]

def load_hidden_states(load_dir, model_name, all_datasets, prefix = "normal", location="auto", layer=-1, dataset_to_prompt_idx = None, num_data = 1000, scale = True, demean = True, mode = "minus", verbose = True):
    if location == "auto":
        location = "decoder" if "gpt" in model_name else "encoder"
    if location == "decoder" and layer < 0:
        layer += models_layer_num[model_name]

    prompt_dict = "ALL" if dataset_to_prompt_idx is None else dataset_to_prompt_idx
    print(f'start loading {location} hidden states {layer} for {model_name} with {prefix} prefix. Prompt_dict: {prompt_dict}, Scale: {scale}, Demean: {demean}, Mode: {mode}')

    if dataset_to_prompt_idx == None:
        # Create a new dictionary where keys are from all_datasets and value is None # WHY None?
        dataset_to_prompt_idx = {key: None for key in all_datasets}

    dataset_to_hiddenstates = {set_name: loadHiddenStates(model_name, set_name, load_dir, dataset_to_prompt_idx[set_name], location, layer, num_data = num_data, confusion = prefix, scale = scale, demean = demean, mode = mode, verbose = verbose) for set_name in all_datasets}
    return dataset_to_hiddenstates

def get_zeros_acc(csv_name, mdl_name, dataset_list, prefix, prompt_dict = None, avg = False):
    """
    Input:
        csv_name: The name of csv we get accuracy from.
        mdl_name: The name of the model.
        dataset_list: List of dataset you want the accuracy from.
        prefix: The name of prefix.
        prompt_dict: Same as in getDir(). You can specify which prompt to get using this variable. Default is None, i.e. get all prompts.
        avg: Whether to average upon return. If True, will return a numbers, otherwise a dict with key from dataset_list and values being a list of accuracy.
    
    Output: 
        dataset_avg: number / dict, depending on `avg` variable.
    """
    
    zeros = pd.read_csv(os.path.join(load_dir, csv_name + ".csv"))
    zeros.dropna(subset=["calibrated"], inplace=True)
    subzeros = zeros.loc[(zeros["model"] == mdl_name) & (zeros["prefix"] == prefix)]
    
    # Extend prompt_dict to ALL dict if it is None
    if prompt_dict is None:
        prompt_dict = {key: range(1000) for key in dataset_list}
    
    # Extract accuracy, each key is a set name and value is a list of acc
    acc_dict = {}
    for dataset in dataset_list:
        filtered_csv = subzeros.loc[(subzeros["dataset"] == dataset) & (
            subzeros["prompt_idx"].isin(prompt_dict[dataset])
        )]
        acc_dict[dataset] = filtered_csv["calibrated"].to_list()

    if not avg:
        return acc_dict
    else:
        # get the dataset avg, and finally the global level avg
        return np.mean([np.mean(values) for values in acc_dict.values()])