from pathlib import Path
import numpy as np
import pandas as pd
import time


def get_directory(
    save_base_dir, model_name, dataset_name_w_num, prefix, token_place, tags=None
):
    """
    Create a directory name given a model, dataset, number of data points and prefix.

    Args:
        save_base_dir (str): the base directory to save the hidden states
        model_name (str): the name of the model
        dataset_name_w_num (str): the name of the dataset with the number of datapoints
        prefix (str): the prefix
        token_place (str): Determine which token's hidden states will be extractd.
            Can be `first` or `last` or `average`
        tags (list): an optional list of strings that describe the hidden state

    Returns:
            directory (Path): the directory
    """
    directory = (
        f"{save_base_dir}/{model_name}_{dataset_name_w_num}_{prefix}_{token_place}"
    )

    if tags is not None:
        for tag in tags:
            directory += f"_{tag}"

    return Path(directory)


def save_hidden_state_to_np_array(hidden_state, dataset_name_w_num, type_list, args):
    """
    Save the hidden states to a numpy array at the directory created by `get_directory`.

    Args:
        hidden_state (list): a list of hidden state arrays
        dataset_name_w_num (str): the name of the dataset with the number of data points
        type_list (list): a list of strings that describe the type of hidden state
        args (argparse.Namespace): the arguments

    Returns:
            None
    """
    directory = get_directory(
        args.save_base_dir,
        args.model,
        dataset_name_w_num,
        args.prefix,
        args.token_place,
    )
    directory.mkdir(parents=True, exist_ok=True)

    # hidden states is num_data * layers * dim
    # logits is num_data * vocab_size
    for (typ, array) in zip(type_list, hidden_state):
        if args.save_all_layers or "logits" in typ:
            np.save(directory / f"{typ}.npy", array)
        else:
            # only save the last layers for encoder hidden states
            for idx in args.states_index:
                np.save(
                    directory / f"{typ}_{args.states_location}{idx}.npy",
                    array[:, idx, :],
                )


def save_records_to_csv(records, args):
    """
    Save the records to a csv file at the base directory + save csv name.

    Args:
        records (list): a list of dictionaries that contains metadata about experiment
        args (argparse.Namespace): the arguments

    Returns:
            None
    """
    file_path = args.save_base_dir / f"{args.save_csv_name}.csv"
    if not file_path.exists():
        all_results = pd.DataFrame(
            columns=[
                "time",
                "model",
                "dataset",
                "prompt_idx",
                "num_data",
                "population",
                "prefix",
                "cal_zeroshot",
                "cal_hiddenstates",
                "log_probs",
                "calibrated",
                "tag",
            ]
        )
    else:
        all_results = pd.read_csv(file_path)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for dic in records:
        dic["time"] = current_time
        spliter = dic["dataset"].split("_")
        dic["dataset"], dic["prompt_idx"] = spliter[0], int(spliter[2][6:])

    all_results = all_results.append(records, ignore_index=True)
    all_results.to_csv(file_path, index=False)

    print(f"Successfully saved {len(records)} items in records to {file_path}")
