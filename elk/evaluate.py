from elk.utils_evaluation.ccs import CCS
from elk.utils_evaluation.utils_evaluation import (
    load_hidden_states,
    get_permutation,
    split,
    append_stats,
)
from elk.utils_evaluation.parser import get_args
from elk.utils_evaluation.utils_evaluation import save_df_to_csv
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch


def evaluate(args, logistic_regression_model, ccs_model: CCS):
    args.save_dir.mkdir(parents=True, exist_ok=True)

    hidden_states = load_hidden_states(
        hidden_states_directory=args.hidden_states_directory,
        model_name=args.model,
        dataset_name=args.dataset,
        prefix=args.prefix,
        language_model_type=args.language_model_type,
        layer=args.layer,
        mode=args.mode,
        num_data=args.num_data,
    )
    permutation = get_permutation(hidden_states)

    accuracies_ccs = []
    losses_ccs = []
    accuracies_lr = []
    losses_lr = []
    for prompt_idx in range(len(hidden_states)):
        features, labels = split(
            hidden_states=hidden_states,
            permutation=permutation,
            prompts=[prompt_idx],
            split="test",
        )

        print("Evaluating logistic regression model")
        acc_lr = logistic_regression_model.score(features, labels)
        accuracies_lr.append(acc_lr)
        losses_lr.append(0)  # TODO: get loss from lr somehow

        print("Evaluating CCS model")
        x0, x1 = torch.from_numpy(features).to(args.device).chunk(2, dim=1)
        labels = torch.from_numpy(labels).to(args.device)
        acc_ccs, loss_ccs = ccs_model.score((x0, x1), labels)
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

    stats_df = pd.DataFrame(
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
    stats_df = append_stats(
        stats_df, args, "ccs", avg_accuracy_ccs, avg_accuracy_std_ccs, avg_loss_ccs
    )
    stats_df = append_stats(
        stats_df, args, "lr", avg_accuracy_lr, avg_accuracy_std_lr, avg_loss_lr
    )
    save_df_to_csv(args, stats_df, args.prefix)


if __name__ == "__main__":
    args = get_args(default_config_path=Path(__file__).parent / "default_config.json")

    with open(args.trained_models_path / "logistic_regression_model.pkl", "rb") as file:
        logistic_regression_model = pickle.load(file)

    ccs_model = CCS.load(args.trained_models_path / "ccs_model.pt")
    evaluate(args, logistic_regression_model, ccs_model)
