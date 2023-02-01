from pathlib import Path
from utils_extraction.func_utils import getAvg, adder
from utils_extraction.load_utils import getDic, set_load_dir, get_zeros_acc
from utils_extraction.method_utils import mainResults
import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import time

# JSON Load
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_models = global_dict["registered_models"]
registered_prefix = global_dict["registered_prefix"]
models_layer_num = global_dict["models_layer_num"]


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=registered_models)
parser.add_argument(
    "--prefix", nargs="+", default=["normal"], choices=registered_prefix
)
parser.add_argument("--datasets", nargs="+", default=registered_dataset_list)
parser.add_argument(
    "--test", type=str, default="testall", choices=["testone", "testall"]
)
parser.add_argument("--data_num", type=int, default=1000)
parser.add_argument(
    "--method_list", nargs="+", default=["0-shot", "TPC", "KMeans", "LR", "BSS", "Prob"]
)
parser.add_argument(
    "--mode",
    type=str,
    default="auto",
    choices=["auto", "minus", "concat"],
    help="How you combine h^+ and h^-.",
)
parser.add_argument(
    "--save_dir",
    type=Path,
    default="extraction_results",
    help="where the csv and params are saved",
)
parser.add_argument(
    "--append",
    action="store_true",
    help="Whether to append content in frame rather than rewrite.",
)
parser.add_argument(
    "--load_dir",
    type=Path,
    default="generation_results",
    help="Where the hidden states and zero-shot accuracy are loaded.",
)
parser.add_argument("--location", type=str, default="auto")
parser.add_argument("--layer", type=int, default=-1)
parser.add_argument("--zero", type=Path, default="generation_results")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--prompt_save_level", default="all", choices=["single", "all"])
args = parser.parse_args()

dataset_list = args.datasets
set_load_dir(args.load_dir)
assert args.test != "testone", NotImplementedError(
    "Current extraction program does not support applying method on prompt-specific"
    " level."
)

if args.location == "auto":
    args.location = "decoder" if "gpt" in args.model else "encoder"
if args.location == "decoder" and args.layer < 0:
    args.layer += models_layer_num[args.model]

print("-------- args --------")
for key in list(vars(args).keys()):
    print("{}: {}".format(key, vars(args)[key]))
print("-------- args --------")


def saveParams(name, coef, intercept):
    path = args.save_dir / "params"
    np.save(path / "coef_{}.npy".format(name), coef)
    np.save(path / "intercept_{}.npy".format(name), intercept)


def saveCsv(csv, prefix, str=""):
    dir = args.save_dir / "{}_{}_{}.csv".format(args.model, prefix, args.seed)
    csv.to_csv(dir, index=False)
    print(
        "{} Saving to {} at {}".format(
            str, dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
    )


if __name__ == "__main__":
    # Make sure output folders exist
    param_dir = args.save_dir / "params"
    args.save_dir.mkdir(exist_ok=True, parents=True)
    param_dir.mkdir(exist_ok=True, parents=True)

    # each loop will generate a csv file
    for global_prefix in args.prefix:
        print(
            "---------------- model = {}, prefix = {} ----------------".format(
                args.model, global_prefix
            )
        )
        # Set the random seed, in which case the permutation_dict will be the same
        random.seed(args.seed)
        np.random.seed(args.seed)

        # shorten the name
        model = args.model
        data_num = args.data_num

        # Start calculate numbers
        # std is over all prompts within this dataset
        if not args.append:
            csv = pd.DataFrame(
                columns=[
                    "model",
                    "prefix",
                    "method",
                    "prompt_level",
                    "train",
                    "test",
                    "accuracy",
                    "std",
                ]
            )
        else:
            dir = args.save_dir / f"{args.model}_{global_prefix}_{args.seed}.csv"
            csv = pd.read_csv(dir)
            print(
                "Loaded {} at {}".format(
                    dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                )
            )

        if "0-shot" in args.method_list:
            # load zero-shot performance
            rawzeros = pd.read_csv(args.load_dir / f"{args.zero}.csv")
            # Get the global zero acc dict (setname, [acc])
            zeros_acc = get_zeros_acc(
                csv_name=args.zero,
                mdl_name=model,
                dataset_list=dataset_list,
                prefix=global_prefix,
            )
            for setname in dataset_list:
                if args.prompt_save_level == "all":
                    csv = adder(
                        csv,
                        model,
                        global_prefix,
                        "0-shot",
                        "",
                        "",
                        setname,
                        np.mean(zeros_acc[setname]),
                        np.std(zeros_acc[setname]),
                        "",
                        "",
                        "",
                    )
                else:  # For each prompt, save one line
                    for idx in range(len(zeros_acc[setname])):
                        csv = adder(
                            csv,
                            model,
                            global_prefix,
                            "0-shot",
                            prompt_level=idx,
                            train="",
                            test=setname,
                            accuracy=zeros_acc[setname][idx],
                            std="",
                            location="",
                            layer="",
                            loss="",
                        )

            saveCsv(csv, global_prefix, "After calculating zeroshot performance.")

        for method in args.method_list:
            if method == "0-shot":
                continue
            print("-------- method = {} --------".format(method))

            mode = (
                args.mode
                if args.mode != "auto"
                else ("minus" if method != "Prob" else "concat")
            )
            # load the data_dict and permutation_dict
            data_dict, permutation_dict = getDic(
                mdl_name=model,
                dataset_list=dataset_list,
                prefix=global_prefix,
                location=args.location,
                layer=args.layer,
                mode=mode,
            )

            test_dict = {key: range(len(data_dict[key])) for key in dataset_list}

            for train_set in ["all"] + dataset_list:
                train_list = dataset_list if train_set == "all" else [train_set]
                projection_dict = {
                    key: range(len(data_dict[key])) for key in train_list
                }

                n_components = 1 if method == "TPC" else -1

                # return a dict with the same shape as test_dict
                # for each key test_dict[key] is a unitary list
                res, lss, pmodel, cmodel = mainResults(
                    data_dict=data_dict,
                    permutation_dict=permutation_dict,
                    projection_dict=projection_dict,
                    test_dict=test_dict,
                    n_components=n_components,
                    projection_method="PCA",
                    classification_method=method,
                )

                # save params except for KMeans
                if method in ["TPC", "BSS"]:
                    coef, bias = cmodel.coef_ @ pmodel.getDirection(), cmodel.intercept_
                    saveParams(
                        "{}_{}_{}_{}_{}_{}".format(
                            model, global_prefix, method, "all", train_set, args.seed
                        ),
                        coef,
                        bias,
                    )

                acc, std, loss = (
                    getAvg(res),
                    np.mean([np.std(lis) for lis in res.values()]),
                    np.mean([np.mean(lis) for lis in lss.values()]),
                )
                print(
                    "method = {:8}, prompt_level = {:8}, train_set = {:20}, avgacc is"
                    " {:.2f}, std is {:.2f}, loss is {:.4f}".format(
                        method, "all", train_set, 100 * acc, 100 * std, loss
                    )
                )

                for key in dataset_list:
                    if args.prompt_save_level == "all":
                        csv = adder(
                            csv,
                            model,
                            global_prefix,
                            method,
                            "all",
                            train_set,
                            key,
                            accuracy=np.mean(res[key]),
                            std=np.std(res[key]),
                            location=args.location,
                            layer=args.layer,
                            loss=np.mean(lss[key]) if method in ["Prob", "BSS"] else "",
                        )
                    else:
                        for idx in range(len(res[key])):
                            csv = adder(
                                csv,
                                model,
                                global_prefix,
                                method,
                                idx,
                                train_set,
                                key,
                                accuracy=res[key][idx],
                                std="",
                                location=args.location,
                                layer=args.layer,
                                loss=lss[key][idx] if method in ["Prob", "BSS"] else "",
                            )

        saveCsv(csv, global_prefix, "After finish {}".format(method))
