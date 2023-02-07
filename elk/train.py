import pickle
import random

import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .eval.utils_evaluation import load_hidden_states
from .files import elk_cache_dir
from .training.ccs import CCS


@torch.autocast("cuda", enabled=torch.cuda.is_available())
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
    assert np.mean(labels) not in [0, 1]

    train_hiddens, _, train_labels, __ = train_test_split(
        hiddens, labels, random_state=args.seed, stratify=labels
    )
    assert isinstance(train_hiddens, torch.Tensor)

    ccs_models = []
    lr_models = []

    pbar = tqdm(train_hiddens.unbind(1), unit="layer")
    for layer_hiddens in pbar:
        # TODO: Once we implement cross-validation for CCS, we should benchmark against
        # LogisticRegressionCV here.
        pbar.set_description("Fitting LR")
        lr_model = LogisticRegression(max_iter=10000, n_jobs=1, C=0.1)
        lr_model.fit(layer_hiddens, train_labels)

        pbar.set_description("Fitting CCS")
        x0, x1 = layer_hiddens.to(args.device).chunk(2, dim=1)

        ccs_model = CCS(in_features=layer_hiddens.shape[1] // 2, device=args.device)
        ccs_model.fit(
            data=(x0, x1),
            optimizer=args.optimizer,
            verbose=False,
            weight_decay=args.weight_decay,
        )

        lr_models.append(lr_model)
        ccs_models.append(ccs_model)

    return lr_models, ccs_models


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

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    lr_models, ccs_models = train(args)

    path = elk_cache_dir() / args.name
    with open(path / "lr_models.pkl", "wb") as file:
        pickle.dump(lr_models, file)

    torch.save(ccs_models, path / "ccs_models.pt")
