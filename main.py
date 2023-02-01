import os
import random

import numpy as np
import pickle

from train import train
from evaluate import evaluate
from utils_evaluation.parser import get_args

if __name__ == "__main__":
    args = get_args(default_config_path = "default_config.json")
    os.makedirs(args.trained_models_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    logistic_regression_model, ccs_model = train(args)
    
    # save models # TODO: use better filename for the pkls, so they don't get overwritten
    with open(os.path.join(args.trained_models_path, "logistic_regression_model.pkl"), 'wb') as file:
        pickle.dump(logistic_regression_model, file)
    with open(os.path.join(args.trained_models_path, "ccs_model.pkl"), 'wb') as file:
        pickle.dump(ccs_model, file)

    evaluate(args, logistic_regression_model, ccs_model)



