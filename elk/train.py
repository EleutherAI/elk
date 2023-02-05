from elk.utils_evaluation.ccs import CCS
from elk.utils_evaluation.utils_evaluation import (
    load_hidden_states,
    get_permutation,
    split,
)
from elk.utils_evaluation.parser import get_args
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import random
import torch


def train(args):
    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Extract hidden states from the model
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

    # `features` is of shape [batch_size, hidden_size * 2]
    # the first half of the features are from the first sentence,
    # the second half are from the second sentence
    features, labels = split(
        hidden_states,
        get_permutation(hidden_states),
        prompts=range(len(hidden_states)),
        split="train",
    )
    assert len(features.shape) == 2

    # TODO: Once we implement cross-validation for CCS, we should benchmark it against
    # LogisticRegressionCV here.
    print("Fitting logistic regression model...")
    logistic_regression_model = LogisticRegression(max_iter=10000, n_jobs=1, C=0.1)
    logistic_regression_model.fit(features, labels)
    print("Done.")

    print("Training CCS model...")
    x0, x1 = torch.from_numpy(features).to(args.device).chunk(2, dim=1)

    ccs_model = CCS(in_features=features.shape[1] // 2, device=args.device)
    ccs_model.fit(
        data=(x0, x1),
        optimizer=args.optimizer,
        verbose=True,
        weight_decay=args.weight_decay,
    )
    print("Done.")

    return logistic_regression_model, ccs_model


if __name__ == "__main__":
    args = get_args(default_config_path=Path(__file__).parent / "default_config.json")
    print(f"-------- args = {args} --------")

    logistic_regression_model, ccs_model = train(args)

    # save models
    # TODO: use better filenames for the pkls, so they don't get overwritten
    args.trained_models_path.mkdir(parents=True, exist_ok=True)
    with open(args.trained_models_path / "logistic_regression_model.pkl", "wb") as file:
        pickle.dump(logistic_regression_model, file)

    ccs_model.save(args.trained_models_path / "ccs_model.pt")
