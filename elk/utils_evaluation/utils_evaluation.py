import os
import numpy as np
import pandas as pd
import json
import time

from pathlib import Path

default_config_path = (
    Path(__file__).parent.parent / Path(__file__).parent.parent / "default_config.json"
)

with open(default_config_path, "r") as f:
    default_config = json.load(f)
datasets = default_config["datasets"]
model_shortcuts = default_config["model_shortcuts"]
prefix = default_config["prefix"]


def get_filtered_filenames(
    directory, model_name, dataset_name, num_data, prefix, place
):
    """
    Returns a list of filenames in the given directory
    that match the specified filter_criteria.

    :param model_name: the name of the model
    :param dataset_name: the name of the dataset
    :param hidden_states_directory: directory containing files
    :param num_data: number of data
    :param confusion: the confusion
    :param place: the place where the data is from
    """
    files = map(lambda path: str(path.relative_to(directory)), directory.glob("**/"))
    filter_criteria = (
        lambda file_name: file_name.startswith(model_name)
        and f"_{dataset_name}_" in file_name
        and f"_{num_data}_" in file_name
        and f"_{prefix}_" in file_name
        and place in file_name
    )
    filtered_files = filter(filter_criteria, files)
    return [directory / f for f in filtered_files]


def organize(hidden_states, mode):
    """
    Whether to do minus, to concat or do nothing
    """
    if mode in ["0", "1"]:
        return hidden_states[int(mode)]
    elif mode == "minus":
        return hidden_states[0] - hidden_states[1]
    elif mode == "concat":
        return np.concatenate(hidden_states, axis=-1)

    raise NotImplementedError("This mode is not supported.")


def normalize(data, scale=True, demean=True):
    # demean the array and rescale each data point
    data = data - np.mean(data, axis=0) if demean else data
    if not scale:
        return data
    norm = np.linalg.norm(data, axis=1)
    avgnorm = np.mean(norm)
    return data / avgnorm * np.sqrt(data.shape[1])


def get_permutation(hidden_states, rate=0.6):
    labels = hidden_states[0][1]  # TODO: Are these really labels?
    labels_length = len(labels)
    permutation = np.random.permutation(range(labels_length)).reshape(-1)
    return [
        permutation[: int(labels_length * rate)],
        permutation[int(labels_length * rate) :],
    ]


def get_hidden_states(
    hidden_states_directory,
    model_name,
    dataset_name,
    prefix="normal",
    language_model_type="encoder",
    layer=-1,
    num_data=1000,
    scale=True,
    demean=True,
    mode="minus",
    place="last",
):
    print(
        f"start loading {language_model_type} hidden states {layer} for"
        f" {model_name} with {prefix} prefix. Scale: {scale}, Demean: {demean}, Mode:"
        f" {mode}"
    )

    filtered_filenames = get_filtered_filenames(
        hidden_states_directory, model_name, dataset_name, num_data, prefix, place
    )
    print("filtered_filenames", filtered_filenames)

    append_list = ["_" + language_model_type + str(layer) for _ in filtered_filenames]

    hidden_states = []
    for filename, append in zip(filtered_filenames, append_list):
        negative_states = np.load(filename / f"0{append}.npy")
        positive_states = np.load(filename / f"1{append}.npy")
        organized_states = organize([negative_states, positive_states], mode=mode)
        hidden_states.append(organized_states)

    # normalize
    normalized_hidden_states = [normalize(w, scale, demean) for w in hidden_states]

    print(
        f"{len(normalized_hidden_states)} prompts for {dataset_name}, with shape"
        f" {normalized_hidden_states[0].shape}"
    )
    labels = [
        np.array(pd.read_csv(filename / "frame.csv")["label"].to_list())
        for filename in filtered_filenames
    ]

    return [(u, v) for u, v in zip(normalized_hidden_states, labels)]


# TODO: Make this function less insane
def split(hidden_states, permutation, prompts, split="train"):
    split_idx = 0 if split == "train" else 1
    split = []
    for prompt_idx in prompts:
        labels = hidden_states[prompt_idx][1]
        split.append(
            [
                hidden_states[prompt_idx][0][permutation[split_idx]],
                labels[permutation[split_idx]],
            ]
        )

    hidden_states = np.concatenate([w[0] for w in split], axis=0)
    labels = np.concatenate([w[1] for w in split], axis=0)

    return hidden_states, labels


def save_df_to_csv(args, df, prefix):
    dir = args.save_dir / f"{args.model}_{prefix}_{args.seed}.csv"
    df.to_csv(dir, index=False)
    print(f"Saving to {dir} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def append_stats(stats_df, args, method, avg_accuracy, avg_accuracy_std, avg_loss):
    return stats_df.append(
        {
            "model": args.model,
            "method": method,
            "prefix": args.prefix,
            "prompt_level": "all",
            # TODO:for now we train and test on the
            # same dataset (with a split of course)
            "train": args.dataset,
            "test": args.dataset,
            "accuracy": avg_accuracy,
            "std": avg_accuracy_std,
            "language_model_type": args.language_model_type,
            "layer": args.layer,
            "loss": avg_loss,
        },
        ignore_index=True,
    )
