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

def get_filtered_filenames(directory, model_name, dataset_name, num_data, confusion, place, prompt_idx=None):
    """
    Returns a list of filenames in the given directory that match the specified filter_criteria.

    :param model_name: the name of the model
    :param dataset_name: the name of the dataset
    :param hidden_states_directory: the directory containing the files
    :param num_data: number of data
    :param confusion: the confusion
    :param place: the place where the data is from
    :param prompt_idx: a list of prompt indices (optional)
    """
    files = os.listdir(directory)
    filter_criteria = (
        lambda file_name: file_name.startswith(model_name) and
                  f"_{dataset_name}_" in file_name and
                  f"_{num_data}_" in file_name and
                  f"_{confusion}_" in file_name and
                  place in file_name
    )
    filtered_files = filter(filter_criteria, files)
    if prompt_idx is not None:     
        filtered_files = [file for file in filtered_files if int(f.split("_")[3][6:]) in prompt_idx]
    return [os.path.join(directory, f) for f in filtered_files]

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
    
    raise NotImplementedError("This mode is not supported.")

def normalize(data, scale =True, demean = True):
    # demean the array and rescale each data point
    data = data - np.mean(data, axis = 0) if demean else data
    if not scale:
        return data
    norm = np.linalg.norm(data, axis=1)
    avgnorm = np.mean(norm)
    return data / avgnorm * np.sqrt(data.shape[1])

def loadHiddenStates(model_name, dataset_name, hidden_states_directory, promtpt_idx, language_model_type = "encoder", layer = -1, num_data = 1000, confusion = "normal", place = "last", scale = True, demean = True, mode = "minus", verbose = True):
    '''
        Load generated hidden states, return a dict where key is the dataset name and values is a list. Each tuple in the list is the (x,y) pair of one prompt.
        if mode == minus, then get h - h'
        if mode == concat, then get np.concatenate([h,h'])
        elif mode == 0 or 1, then get h or h'
    '''

    filtered_filenames = get_filtered_filenames(hidden_states_directory, model_name, dataset_name, num_data, confusion, place, promtpt_idx)
    print("filtered_filenames", filtered_filenames)
    
    append_list = ["_" + language_model_type + str(layer) for _ in filtered_filenames]

    hidden_states = []
    for filename, app in zip(filtered_filenames, append_list):
        negative_states = np.load(os.path.join(filename, f"0{app}.npy"))
        positive_states = np.load(os.path.join(filename, f"1{app}.npy"))
        organized_states = organizeStates([negative_states, positive_states], mode = mode)
        hidden_states.append(organized_states)

    # normalize
    hidden_states = [normalize(w, scale, demean) for w in hidden_states]
    if verbose:        
        print("{} prompts for {}, with shape {}".format(len(hidden_states), dataset_name, hidden_states[0].shape))
    labels = [np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list()) for w in filtered_filenames]

    return [(u,v) for u,v in zip(hidden_states, labels)]


#TODO: UNDESTSTAND THIS 
def getPermutation(data_list, rate = 0.6):
    length = len(data_list[0][1])
    permutation = np.random.permutation(range(length)).reshape(-1)
    return [permutation[: int(length * rate)], permutation[int(length * rate):]]

def load_hidden_states(hidden_states_directory, model_name, all_datasets, prefix = "normal", language_model_type="encoder", layer=-1, dataset_to_prompt_idx = None, num_data = 1000, scale = True, demean = True, mode = "minus", verbose = True):
    if language_model_type == "decoder" and layer < 0:
        layer += models_layer_num[model_name]

    prompt_dict = "ALL" if dataset_to_prompt_idx is None else dataset_to_prompt_idx
    print(f'start loading {language_model_type} hidden states {layer} for {model_name} with {prefix} prefix. Prompt_dict: {prompt_dict}, Scale: {scale}, Demean: {demean}, Mode: {mode}')

    if dataset_to_prompt_idx == None:
        # Create a new dictionary where keys are from all_datasets and value is None # WHY None?
        dataset_to_prompt_idx = {key: None for key in all_datasets}

    dataset_to_hiddenstates = {dataset_name: loadHiddenStates(model_name, dataset_name, hidden_states_directory, dataset_to_prompt_idx[dataset_name], language_model_type, layer, num_data = num_data, confusion = prefix, scale = scale, demean = demean, mode = mode, verbose = verbose) for dataset_name in all_datasets}
    return dataset_to_hiddenstates