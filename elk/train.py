import pickle
import random

import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .eval.utils_evaluation import load_hidden_states
from .files import elk_cache_dir
from .training.ccs import CCS


def train(args):
    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the hidden states extracted from the model
    hiddens, labels = load_hidden_states(
        path=elk_cache_dir() / args.name / "hiddens.pt",
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
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--device",
        type=str,
        help="PyTorch device to use. Default is cuda:0 if available.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="concat",
        choices=["minus", "concat"],
        help="How you combine h^+ and h^-.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "lbfgs"),
        help="Optimizer for CCS. Should be adam or lbfgs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help=(
            "Weight decay for CCS when using Adam. Used as L2 regularization in LBFGS."
        ),
    )
    args = parser.parse_args()

    lr_model, ccs_model = train(args)

    path = elk_cache_dir() / args.name
    with open(path / "lr_model.pkl", "wb") as file:
        pickle.dump(lr_model, file)

    ccs_model.save(path / "ccs_model.pt")
