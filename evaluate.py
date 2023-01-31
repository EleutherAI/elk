import os
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from utils_extraction.ccs import CCS
from utils_extraction.load_utils import get_hidden_states, get_permutation
from utils_extraction.parser import get_args
from utils_extraction.save_utils import save_df_to_csv


def create_results_dirs(save_dir):
    path=os.path.join(save_dir, "params")
    os.makedirs(path, exist_ok=True)
    print(f"created directories for saving evaluation results: {path}.")

def split(hidden_states, permutation, prompts, split = "train"):
    split_idx = 0 if split == "train" else 1
    split = []
    for prompt_idx in prompts:
        labels = hidden_states[prompt_idx][1]
        split.append([
            hidden_states[prompt_idx][0][permutation[split_idx]],
            labels[permutation[split_idx]]
        ])
    
    hidden_states = np.concatenate([w[0] for w in split], axis=0)
    labels = np.concatenate([w[1] for w in split], axis=0)

    return hidden_states, labels

def append_stats(stats_df, args, method, avg_accuracy, avg_accuracy_std, avg_loss):
    return stats_df.append({
        "model": args.model, 
        "method": method,
        "prefix": args.prefix,
        "prompt_level": "all", 
        "train": args.dataset, # for now we train and test on the same dataset (with a split of course) 
        "test": args.dataset, 
        "accuracy": avg_accuracy,
        "std": avg_accuracy_std,
        "language_model_type": args.language_model_type, 
        "layer": args.layer,
        "loss": avg_loss
    }, ignore_index=True)

if __name__ == "__main__":
    args = get_args(default_config_path = "default_config.json")
    print(f"-------- args = {args} --------")

    create_results_dirs(save_dir=args.save_dir)

    hidden_states = get_hidden_states(
        hidden_states_directory=args.hidden_states_directory,
        model_name=args.model,
        dataset_name=args.dataset,
        prefix=args.prefix,
        language_model_type=args.language_model_type,
        layer=args.layer,
        mode=args.mode,
        num_data=args.num_data
    )

    # Set the random seed for the permutation
    random.seed(args.seed)
    np.random.seed(args.seed)
    permutation = get_permutation(hidden_states)
    print("permutation", permutation)
    data, labels = split(hidden_states, permutation, prompts=range(len(hidden_states)), split="test")
    assert len(data.shape) == 2

    # Train classification model
    print("train classification model")
    logistic_regression_model = LogisticRegression(max_iter = 10000, n_jobs = 1, C = 0.1)
    logistic_regression_model.fit(data, labels)
    print("done training classification model")

    # Train ccs model
    print("train ccs model")
    ccs = CCS(verbose=True)    
    data = [data[:,:data.shape[1]//2], data[:,data.shape[1]//2:]]
    ccs.fit(data = data, label=labels, device=args.model_device) 
    print("done training ccs model") 
    
    accuracies_ccs = []
    losses_ccs = []    
    accuracies_lr = []
    losses_lr = []
    for prompt_idx in range(len(hidden_states)):
        data, labels = split(hidden_states=hidden_states, permutation=permutation,  prompts=[prompt_idx], split="test")

        # evaluate classification model
        print("evaluate classification model")
        acc_lr = logistic_regression_model.score(data, labels)
        accuracies_lr.append(acc_lr)
        losses_lr.append(0) # TODO: get loss from lr somehow
        
        # evaluate ccs model
        print("evaluate ccs model")
        data = [data[:,:data.shape[1]//2], data[:,data.shape[1]//2:]]
        acc_ccs, loss_ccs = ccs.score(data, labels, getloss = True)
        accuracies_ccs.append(acc_ccs)
        losses_ccs.append(loss_ccs)       

    avg_accuracy_ccs = np.mean(accuracies_ccs)
    avg_accuracy_std_ccs = np.std(accuracies_ccs)
    avg_loss_ccs = np.mean(losses_ccs)
    
    avg_accuracy_lr = np.mean(accuracies_lr)
    avg_accuracy_std_lr = np.std(accuracies_lr)
    avg_loss_lr = np.mean(losses_lr)

    print("avg_accuracy_ccs", avg_accuracy_ccs)
    print("avg_accuracy_std_ccs", avg_accuracy_std_ccs)
    print("avg_loss_ccs", avg_loss_ccs)

    print("avg_accuracy_lr", avg_accuracy_lr)
    print("avg_accuracy_std_lr", avg_accuracy_std_lr)
    print("avg_loss_lr", avg_loss_lr)
    
    stats_df = pd.DataFrame(columns = ["model", "prefix", "method", "prompt_level", "train", "test", "accuracy", "std"])
    stats_df = append_stats(stats_df, args, "ccs", avg_accuracy_ccs, avg_accuracy_std_ccs, avg_loss_ccs)
    stats_df = append_stats(stats_df, args, "lr", avg_accuracy_lr, avg_accuracy_std_lr, avg_loss_lr)
    save_df_to_csv(args, stats_df, args.prefix, f"After finish")