import csv
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
    cache_dir = elk_cache_dir() / args.name
    hiddens, labels = load_hidden_states(
        path=cache_dir / "hiddens.pt",
        reduce=args.mode,
    )
    assert np.mean(labels) not in [0, 1]

    train_hiddens, test_hiddens, train_labels, test_labels = train_test_split(
        hiddens, labels, random_state=args.seed, stratify=labels
    )
    assert isinstance(test_hiddens, torch.Tensor)
    assert isinstance(train_hiddens, torch.Tensor)

    ccs_models = []
    lr_models = []
    L = train_hiddens.shape[1]

    # Do the last layer first- useful for debugging, maybe change later
    test_layers = list(test_hiddens.unbind(1))
    train_layers = list(train_hiddens.unbind(1))
    test_layers.reverse()
    train_layers.reverse()

    pbar = tqdm(zip(train_layers, test_layers), total=L, unit="layer")
    writer = csv.writer(open(cache_dir / "eval.csv", "w"))
    writer.writerow(["layer", "train_loss", "test_loss", "test_acc", "lr_acc"])

    for train_h, test_h in pbar:
        # TODO: Once we implement cross-validation for CCS, we should benchmark against
        # LogisticRegressionCV here.
        pbar.set_description("Fitting LR")
        lr_model = LogisticRegression(max_iter=10000, n_jobs=1, C=0.1)
        lr_model.fit(train_h, train_labels)
        lr_acc = lr_model.score(test_h, test_labels)

        pbar.set_description("Fitting CCS")
        x0, x1 = train_h.to(args.device).chunk(2, dim=-1)
        test_x0, test_x1 = test_h.to(args.device).chunk(2, dim=-1)

        ccs_model = CCS(in_features=x0.shape[-1], device=args.device)
        train_loss = ccs_model.fit(
            data=(x0, x1),
            optimizer=args.optimizer,
            verbose=False,
            weight_decay=args.weight_decay,
        )
        test_acc, test_loss = ccs_model.score(
            (test_x0, test_x1), torch.tensor(test_labels, device=args.device)
        )
        pbar.set_postfix(loss=train_loss, ccs_acc=test_acc, lr_acc=lr_acc)
        writer.writerow([L - pbar.n, train_loss, test_loss, test_acc, lr_acc])

        lr_models.append(lr_model)
        ccs_models.append(ccs_model)

    ccs_models.reverse()
    lr_models.reverse()
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
