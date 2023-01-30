import os 
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from utils_extraction.load_utils import load_hidden_states, get_permutation
from utils_extraction.method_utils import get_main_results
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
        # Set the random seed, in which case the permutation will be the same
        random.seed(args.seed)
        np.random.seed(args.seed)

        stats_df = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])

        # TODO Remove for loop over methods
        for method in tqdm(args.methods, desc='Iterating over classification methods:', position=1, leave=False):
            print(f"-------- method = {method} --------")

            mode = args.mode
            if args.mode == "auto":
                # overwrite mode if set to auto
                mode = "concat" if method == "Prob" else "minus"

            hidden_states = load_hidden_states(
                hidden_states_directory=args.hidden_states_directory,
                model_name= args.model, 
                dataset_name=args.dataset,
                prefix = prefix,
                language_model_type = args.language_model_type,
                layer = args.layer,
                mode = mode,
                num_data = args.num_data
            )

            permutation = get_permutation(hidden_states)
            print("permutation", permutation)

            test = range(len(hidden_states))
            print("test", test)

            accuraries_per_prompt, losses_per_prompt = get_main_results(
                hidden_states = hidden_states, 
                permutation = permutation, 
                test = test, 
                n_components = -1, 
                projection_method = "PCA",
                classification_method = method,
                device = args.model_device)

            avg_accuracy = np.mean(accuraries_per_prompt)
            avg_accuracy_std = np.std(accuraries_per_prompt)
            avg_loss = np.mean(losses_per_prompt)

            stats_df = stats_df.append({
                "model": args.model, 
                "prefix": prefix,
                "method": method, 
                "prompt_level": "all", 
                "train": args.dataset, # for now we train and test on the same dataset (with a split of course) 
                "test": args.dataset, 
                "accuracy": avg_accuracy,
                "std": avg_accuracy_std,
                "language_model_type": args.language_model_type, 
                "layer": args.layer,
                "loss": avg_loss
            }, ignore_index=True)

            save_df_to_csv(args, stats_df, prefix, f"After finish {method}")

    