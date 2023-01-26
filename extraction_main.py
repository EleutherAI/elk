import os 
import pandas as pd
import numpy as np
import time
from utils_extraction.load_utils import get_hiddenstates_and_permutations, set_load_dir
from utils_extraction.method_utils import mainResults
from utils_extraction.func_utils import getAvg, adder
from utils_extraction.parser import get_extraction_args
from utils_extraction.save_utils import save_csv, save_params

import pandas as pd
import random 


args = get_extraction_args(json_dir = "./registration")
dataset_list = args.datasets
set_load_dir(args.load_dir)


print("-------- args --------")
for key in list(vars(args).keys()):
    print("{}: {}".format(key, vars(args)[key]))
print("-------- args --------")

if __name__ == "__main__":
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
            csv = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])
        else:
            dir = os.path.join(args.save_dir, f"{args.model}_{prefix}_{args.seed}.csv")
            csv = pd.read_csv(dir)
            print(f"Loaded {dir} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        for method in args.method_list:
            print("-------- method = {method} --------")

            mode = args.mode if args.mode != "auto" else (
                "minus" if method != "Prob" else "concat"
            )
            dataset_to_hiddenstates, permutation_dict = get_hiddenstates_and_permutations(
                mdl_name= model, 
                all_datasets=dataset_list,
                prefix = prefix,
                location = args.location,
                layer = args.layer,
                mode = mode,
                num_data = num_data
            )

            test_dict = {key: range(len(dataset_to_hiddenstates[key])) for key in dataset_list}

            for train_set in ["all"] + dataset_list:

                train_list = dataset_list if train_set == "all" else [train_set]
                projection_dict = {key: range(len(dataset_to_hiddenstates[key])) for key in train_list}

                n_components = -1

                # return a dict with the same shape as test_dict
                # for each key test_dict[key] is a unitary list
                res, lss, pmodel, cmodel = mainResults(
                    data_dict = dataset_to_hiddenstates, 
                    permutation_dict = permutation_dict, 
                    projection_dict = projection_dict,
                    test_dict = test_dict, 
                    n_components = n_components, 
                    projection_method = "PCA",
                    classification_method = method,
                    device = args.model_device)

                acc, std, loss = getAvg(res), np.mean([np.std(lis) for lis in res.values()]), np.mean([np.mean(lis) for lis in lss.values()])
                print(f"""
                    method = {method}, 
                    prompt_level = {'all'}, train_set = {train_set}, avgacc is {100 * acc}, std is {100 * std}, loss is {loss}
                    """)

                for key in dataset_list:
                    if args.prompt_save_level == "all":
                        csv = adder(csv, model, prefix, method, "all", train_set, key,
                                    accuracy = np.mean(res[key]),
                                    std = np.std(res[key]),
                                    location = args.location,
                                    layer = args.layer,
                                    loss = np.mean(lss[key]) if method in ["Prob", "BSS"] else ""
                                    )
                    else:
                        for idx in range(len(res[key])):
                            csv = adder(csv, model, prefix, method, idx, train_set, key,
                                        accuracy = res[key][idx],
                                        std = "",
                                        location = args.location,
                                        layer = args.layer,
                                        loss = lss[key][idx] if method in ["Prob", "BSS"] else ""
                                        )

        save_csv(args, csv, prefix, f"After finish {method}")

    