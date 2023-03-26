import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import yaml


def get_metric_across_layers(metric_name, reporter_name, reporter_dir, save_csv=False):
    """
    Gets the metric score for a specific model/dataset across layers

    Inputs:
        ``metric_name``: Name of the metric
        ``reporter_name``: Name of 'model_name--dataset_name' to get data from
        ``reporter_dir``: Name of the directory to look for reporter data
        ``save_csv``: Boolean to optionally save output as a csv file

    Outputs:
        ``values``: A pandas series that has an index of layer_num and value of metric
    """
    REPORTER_DIR = os.path.abspath(reporter_dir)

    if not os.path.exists(REPORTER_DIR):
        # Make sure to run this file in the spar-elk-w folder
        raise Exception(f"The directory '{REPORTER_DIR}' does not exist")
    elif not os.path.isdir(REPORTER_DIR):
        raise Exception(f"{REPORTER_DIR} is not a directory")

    reporter_names = os.listdir(REPORTER_DIR)
    reporter_paths = []
    for f in reporter_names:
        path = os.path.join(REPORTER_DIR, f)
        if os.path.isdir(path) and f == reporter_name:
            reporter_paths.append(os.path.join(REPORTER_DIR, f))

    assert len(reporter_paths) == 1, (
        f"There is either no file with the name '{reporter_name}', "
        "or multiple with same name "
        f"The current reporter_paths found are '{reporter_paths}'"
    )

    reporter_path = reporter_paths[0]

    eval_path = os.path.join(reporter_path, "eval.csv")
    data = pd.read_csv(eval_path)

    score_across_layers = data.loc[:, metric_name]

    if save_csv:
        print(os.path.join(reporter_path, f"{metric_name}.csv"))
        score_across_layers.to_csv(os.path.join(reporter_path, f"{metric_name}.csv"))

    return score_across_layers
