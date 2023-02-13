import csv
import pickle
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from elk.files import elk_cache_dir
from elk.training.ccs import CCS
from elk.training.preprocessing import load_hidden_states, normalize
from elk.extensions.recursive_ccs.rccs import RecursiveCCS
from elk.extensions.recursive_ccs.parser import get_rccs_training_parser


@torch.autocast("cuda", enabled=torch.cuda.is_available())
def train(args):
    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the hidden states extracted from the model
    cache_dir = elk_cache_dir() / args.name
    train_hiddens, train_labels = load_hidden_states(
        path=cache_dir / "train_hiddens.pt"
    )
    val_hiddens, val_labels = load_hidden_states(
        path=cache_dir / "validation_hiddens.pt"
    )
    assert len(set(train_labels)) > 1
    assert len(set(val_labels)) > 1

    assert isinstance(val_hiddens, torch.Tensor)
    assert isinstance(train_hiddens, torch.Tensor)

    train_hiddens, val_hiddens = normalize(
        train_hiddens, val_hiddens, args.normalization
    )

    rccs_models = []
    L = train_hiddens.shape[1]

    # Do the last layer first- useful for debugging, maybe change later
    val_layers = list(val_hiddens.unbind(1))
    train_layers = list(train_hiddens.unbind(1))
    val_layers.reverse()
    train_layers.reverse()

    pbar = tqdm(zip(train_layers, val_layers), total=L, unit="layer")
    writer = csv.writer(open(cache_dir / "eval.csv", "w"))
    writer.writerow(
        [
            "layer",
            "iteration",
            "train_loss",
            "loss",
            "acc",
            "cal_acc",
            "auroc",
        ]
    )

    for train_h, val_h in pbar:
        x0, x1 = train_h.to(args.device).chunk(2, dim=-1)
        val_x0, val_x1 = val_h.to(args.device).chunk(2, dim=-1)

        rccs = RecursiveCCS(device=args.device)

        n_probes = 2
        for i in range(n_probes):
            pbar.set_description(f"Fitting RCCS it={i}")
            probe, train_loss = rccs.fit_next_probe(
                data=(x0, x1),
                ccs_params={
                    "device": args.device,
                    "init": args.init,
                    "loss": args.loss,
                },
                train_params={
                    "num_tries": args.num_tries,
                    "optimizer": args.optimizer,
                    "weight_decay": args.weight_decay,
                },
            )
            val_result = probe.score(
                (val_x0, val_x1),
                torch.tensor(val_labels, device=args.device),
            )
            pbar.set_postfix(ccs_auroc=val_result.auroc, it=i)

            stats = [train_loss, *val_result]
            writer.writerow([L - pbar.n] + [i] + [f"{s:.4f}" for s in stats])

        rccs_models.append(rccs.probes)

    rccs_models.reverse()

    path = elk_cache_dir() / args.name
    torch.save(rccs_models, path / "rccs_models.pt")


if __name__ == "__main__":
    args = get_rccs_training_parser().parse_args()
    train(args)
