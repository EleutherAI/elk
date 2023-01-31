import os
import numpy as np
import pandas as pd
import json

######## Load default_config ########
default_config_path = "default_config.json"

with open(default_config_path, "r") as f:
    default_config = json.load(f)
datasets = default_config["datasets"]
models = default_config["models"]
prefix = default_config["prefix"]
models_layer_num = default_config["models_layer_num"]

def get_filtered_filenames(directory, model_name, dataset_name, num_data, prefix, place):
    """
    Returns a list of filenames in the given directory that match the specified filter_criteria.

    :param model_name: the name of the model
    :param dataset_name: the name of the dataset
    :param hidden_states_directory: the directory containing the files
    :param num_data: number of data
    :param confusion: the confusion
    :param place: the place where the data is from
    """
    files = os.listdir(directory)
    filter_criteria = (
        lambda file_name: file_name.startswith(model_name) and
                  f"_{dataset_name}_" in file_name and
                  f"_{num_data}_" in file_name and
                  f"_{prefix}_" in file_name and
                  place in file_name
    )
    filtered_files = filter(filter_criteria, files)
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

def get_permutation(hidden_states, rate = 0.6):
    labels = hidden_states[0][1] # TODO: Are these really labels?
    labels_length = len(labels)
    permutation = np.random.permutation(range(labels_length)).reshape(-1)
    return [permutation[:int(labels_length * rate)], permutation[int(labels_length * rate):]]

def get_hidden_states(hidden_states_directory, model_name, dataset_name, prefix = "normal", language_model_type="encoder", layer=-1, num_data = 1000, scale = True, demean = True, mode = "minus", place = "last"):
    if language_model_type == "decoder" and layer < 0:
        layer += models_layer_num[model_name]

    print(f'start loading {language_model_type} hidden states {layer} for {model_name} with {prefix} prefix. Scale: {scale}, Demean: {demean}, Mode: {mode}')

    filtered_filenames = get_filtered_filenames(hidden_states_directory, model_name, dataset_name, num_data, prefix, place)
    print("filtered_filenames", filtered_filenames)
    
    append_list = ["_" + language_model_type + str(layer) for _ in filtered_filenames]

    hidden_states = []
    for filename, app in zip(filtered_filenames, append_list):
        negative_states = np.load(os.path.join(filename, f"0{app}.npy"))
        positive_states = np.load(os.path.join(filename, f"1{app}.npy"))
        organized_states = organizeStates([negative_states, positive_states], mode = mode)
        hidden_states.append(organized_states)

    # normalize
    normalized_hidden_states = [normalize(w, scale, demean) for w in hidden_states]
    print(f"{len(normalized_hidden_states)} prompts for {dataset_name}, with shape {normalized_hidden_states[0].shape}")
    labels = [np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list()) for w in filtered_filenames]

    return [(u,v) for u,v in zip(normalized_hidden_states, labels)]
        