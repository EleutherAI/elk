import os 
import pandas as pd
import numpy as np
import time
from utils_extraction.load_utils import get_hiddenstates_and_permutations
from utils_extraction.method_utils import get_main_results
from utils_extraction.func_utils import getAvg, populate_stats_df
from utils_extraction.parser import get_extraction_args
from utils_extraction.save_utils import save_df_to_csv, save_params

import pandas as pd
import random 


if __name__ == "__main__":

    args = get_extraction_args(json_dir = "./registration")
    dataset_names = args.datasets
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "params")):
        os.mkdir(os.path.join(args.save_dir, "params"))


    # each loop will generate a csv file
    for prefix in args.prefix:
        print(f"---------------- model = {args.model}, prefix = {prefix} ----------------")
        # Set the random seed, in which case the permutation_dict will be the same
        random.seed(args.seed)
        np.random.seed(args.seed)

        # shorten the name
        model = args.model
        num_data = args.num_data
    
        # Start calculate numbers
        # std is over all prompts within this dataset
        if not args.append:
            stats_df = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])
        else:
            dir = os.path.join(args.save_dir, f"{args.model}_{prefix}_{args.seed}.csv")
            stats_df = pd.read_csv(dir)

        for method in args.method_list:
            print("-------- method = {method} --------")

            # TODO: Write this more elegantly
            mode = args.mode if args.mode != "auto" else (
                "minus" if method != "Prob" else "concat"
            )

            #TODO: UNDERSTAND AND RENAME PERMUTATION DICT
            dataset_to_hiddenstates, permutation_dict = get_hiddenstates_and_permutations(
                load_dir=args.load_dir,
                mdl_name= model, 
                all_datasets=dataset_names,
                prefix = prefix,
                location = args.location,
                layer = args.layer,
                mode = mode,
                num_data = num_data
            )
            
            # TODO: WHY IS THE CONSTRUCTED LIKE THIS AND WHAT IS IT USED FOR?
            test_dict = {dataset: range(len(dataset_to_hiddenstates[dataset])) for dataset in dataset_names}

            for train_set in ["all"] + dataset_names:

                dataset_names_train = dataset_names if train_set == "all" else [train_set]
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

                stats_df = populate_stats_df(dataset_to_accurary_per_prompt, dataset_to_loss_per_prompt, stats_df, model, prefix, method, args, train_set, dataset_names)

            save_df_to_csv(args, stats_df, prefix, f"After finish {method}")

    