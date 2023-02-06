import pytest
import pandas as pd
import pickle
from pathlib import Path

# import train and evaluate
from elk.train import train
from elk.evaluate import evaluate
from elk.utils_evaluation.ccs import CCS
import json

default_config_path = Path(__file__).parent.parent / "default_config.json"

with open(default_config_path, "r") as f:
    default_config = json.load(f)
datasets = default_config["datasets"]
models = default_config["models"]
prefix = default_config["prefix"]


def test_model_performance(model_name, prefix, epsilon, num_data, seed):
    """
    TODO: implement evaluation of the model with all datasets
    # in progress
    """
    hidden_states_directory = Path("../generation_results/test")
    save_dir = Path("../evaluation_results/test")

    for dataset_train in datasets:
        logistic_regression_model, ccs_model = train(
            model=model_name,
            dataset=dataset_train,
            prefix=prefix,
            num_data=num_data,
            seed=seed,
            hidden_states_directory=hidden_states_directory,
            language_model_type="encoder",
            layer=-1,
            mode="minus",
            num_data=1000,
            device="cuda",
            optimizer="adam"
            weight_decay=0.01
        )

        # Get transfer accuracy: evaluate the model on each dataset
        for dataset_eval in datasets:
            evaluate(
                model=model_name,
                dataset=dataset_eval,
                prefix=prefix,
                num_data=num_data,
                seed=seed,
                save_dir=save_dir / f"{model_name}_{dataset_eval}_{prefix}",
                hidden_states_directory=hidden_states_directory,
                logistic_regression_model=logistic_regression_model,
                ccs_model=ccs_model,
            )
        original_eval = pd.read_csv(f"../results.csv")
        current_eval = pd.read_csv(
            f"../evaluation_results/test/{model_name}_{dataset_eval}_"
            f"{prefix}/{model_name}_{prefix}_{seed}.csv"
        )
        compare_df_performance(original_eval, current_eval, epsilon)


def compare_df_performance(original_eval, current_eval, epsilon):
    # TODO: implement the correct format of the csv file
    # make sure current performance is within threshold (epsilon) or better
    for index, row in original_eval.iterrows():
        assert row["accuracy"] - epsilon <= current_eval.iloc[index]["accuracy"]
        assert row["std"] - epsilon <= current_eval.iloc[index]["std"]


def test_deberta():
    test_model_performance(
        model_name="deberta-v2-xxlarge-mnli", 
        prefix="normal", 
        epsilon=3, 
        num_data=1000, 
        seed=0
    )

def test_all_normal_performance():
    for model_name in models:
        test_model_performance(
            model_name, prefix="normal", epsilon=3, num_data=1000, seed=0
        )
