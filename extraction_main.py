import os 
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from utils_extraction.load_utils import load_hidden_states, get_permutation
from utils_extraction.method_utils import get_main_results
from utils_extraction.func_utils import populate_evaluation_results
from utils_extraction.parser import get_args
from utils_extraction.save_utils import save_df_to_csv

import pandas as pd
import random 

def create_results_dirs(save_dir):
    path=os.path.join(save_dir, "params")
    os.makedirs(path, exist_ok=True)
    print(f"created directories for saving evaluation results: {path}.")

if __name__ == "__main__":
    args = get_args(default_config_path = "default_config.json")
    print(f"-------- args = {args} --------")

    create_results_dirs(save_dir=args.save_dir)
    stats_df = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])

    for prefix in tqdm(args.prefix, desc='Iterating over prefixes:', position=0):
        print(f"---------------- model = {args.model}, prefix = {prefix} ----------------")
        # Set the random seed, in which case the permutation_dict will be the same
        random.seed(args.seed)
        np.random.seed(args.seed)

        stats_df = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])

        for method in tqdm(args.methods, desc='Iterating over classification methods:', position=1, leave=False):
            print(f"-------- method = {method} --------")

            mode = args.mode
            if args.mode == "auto":
                # overwrite mode if set to auto
                mode = "concat" if method == "Prob" else "minus"

            hidden_states = load_hidden_states(
                hidden_states_directory=args.hidden_states_directory,
                model_name= args.model, 
                dataset_name=args.datasets[0],
                prefix = prefix,
                language_model_type = args.language_model_type,
                layer = args.layer,
                mode = mode,
                num_data = args.num_data
            )

            dataset_to_hiddenstates = {}
            dataset_to_hiddenstates[args.datasets[0]] = hidden_states

            permutation_dict = {set_name: get_permutation(dataset_to_hiddenstates[set_name]) for set_name in args.datasets}

            # TODO: WHY IS THE CONSTRUCTED LIKE THIS AND WHAT IS IT USED FOR?
            test_dict = {dataset: range(len(dataset_to_hiddenstates[dataset])) for dataset in args.datasets}

            for train_set in tqdm(["all"] + args.datasets, desc='Iterating over train sets:', position=2, leave=False):

                dataset_names_train = args.datasets if train_set == "all" else [train_set]
                #TODO: WHY IS THIS CALLED PROJECTION DICT?
                projection_dict = {dataset: range(len(dataset_to_hiddenstates[dataset])) for dataset in dataset_names_train}

                # return a dict with the same shape as test_dict
                # for each key test_dict[key] is a unitary list
                # TODO: REFACTOR THIS FUNCTION
                dataset_to_accurary_per_prompt, dataset_to_loss_per_prompt = get_main_results(
                    data_dict = dataset_to_hiddenstates, 
                    permutation_dict = permutation_dict, 
                    projection_dict = projection_dict,
                    test_dict = test_dict, 
                    n_components = -1, 
                    projection_method = "PCA",
                    classification_method = method,
                    device = args.model_device)

                avg_accuracy = np.mean([np.mean(accuracies) for accuracies in dataset_to_accurary_per_prompt.values()]) 
                avg_accuracy_std = np.mean([np.std(accuracies) for accuracies in dataset_to_accurary_per_prompt.values()]), 
                avg_loss = np.mean([np.mean(losses) for losses in dataset_to_loss_per_prompt.values()])

                stats_df = populate_evaluation_results(dataset_to_accurary_per_prompt, dataset_to_loss_per_prompt, stats_df, args.model, prefix, method, args, train_set, args.datasets)

            save_df_to_csv(args, stats_df, prefix, f"After finish {method}")

    