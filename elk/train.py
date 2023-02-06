from .eval.parser import get_args
from .eval.utils_evaluation import load_hidden_states
from .training.ccs import CCS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    hiddens, labels = load_hidden_states(
        args.name,
        reduce=args.mode,
    )

    train_hiddens, _, train_labels, __ = train_test_split(
        hiddens, labels, random_state=args.seed, stratify=labels
    )
    assert isinstance(train_hiddens, torch.Tensor)

    # TODO: Once we implement cross-validation for CCS, we should benchmark it against
    # LogisticRegressionCV here.
    print("Fitting logistic regression model...")
    lr_model = LogisticRegression(max_iter=10000, n_jobs=1, C=0.1)
    lr_model.fit(train_hiddens, train_labels)
    print("Done.")

    print("Training CCS model...")
    x0, x1 = train_hiddens.to(args.device).chunk(2, dim=1)

    ccs_model = CCS(in_features=train_hiddens.shape[1] // 2, device=args.device)
    ccs_model.fit(
        data=(x0, x1),
        optimizer=args.optimizer,
        verbose=True,
        weight_decay=args.weight_decay,
    )
    print("Done.")

    return lr_model, ccs_model


if __name__ == "__main__":
    args = get_args()
    lr_model, ccs_model = train(args)

    # save models
    # TODO: use better filenames for the pkls, so they don't get overwritten
    args.trained_models_path.mkdir(parents=True, exist_ok=True)
    with open(args.trained_models_path / "lr_model.pkl", "wb") as file:
        pickle.dump(lr_model, file)

    ccs_model.save(args.trained_models_path / "ccs_model.pt")
