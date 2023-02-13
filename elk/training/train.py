from ..files import elk_cache_dir
from .ccs import CCS
from .parser import get_training_parser
from .preprocessing import load_hidden_states, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
import csv
import numpy as np
import pickle
import random
import torch


# @torch.autocast("cuda", enabled=torch.cuda.is_available())
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

    ccs_models = []
    lr_models = []
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
            "train_loss",
            "loss",
            "acc",
            "cal_acc",
            "auroc",
            "lr_auroc",
            "lr_acc",
        ]
    )

    for train_h, val_h in pbar:
        x0, x1 = train_h.to(args.device).float().chunk(2, dim=-1)
        val_x0, val_x1 = val_h.to(args.device).float().chunk(2, dim=-1)

        train_labels_aug = train_labels + [1 - label for label in train_labels]
        val_labels_aug = val_labels + [1 - label for label in val_labels]

        pbar.set_description("Fitting CCS")
        ccs_model = CCS(
            in_features=x0.shape[-1], device=args.device, init=args.init, loss=args.loss
        )
        if args.label_frac:
            num_labels = round(args.label_frac * len(train_labels))
            labels = torch.tensor(train_labels[:num_labels], device=args.device)
        else:
            labels = None

        train_loss = ccs_model.fit(
            contrast_pair=(x0, x1),
            labels=labels,
            num_tries=args.num_tries,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
        )
        val_result = ccs_model.score(
            (val_x0, val_x1),
            torch.tensor(val_labels, device=args.device),
        )
        pbar.set_postfix(train_loss=train_loss, ccs_auroc=val_result.auroc)

        # TODO: Once we implement cross-validation for CCS, we should benchmark against
        # LogisticRegressionCV here.
        pbar.set_description("Fitting LR")
        lr_model = LogisticRegression(max_iter=10_000)
        lr_model.fit(torch.cat([x0, x1]).cpu(), train_labels_aug)

        lr_preds = lr_model.predict_proba(torch.cat([val_x0, val_x1]).cpu())[:, 1]
        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)
        pbar.set_postfix(
            train_loss=train_loss, ccs_auroc=val_result.auroc, lr_auroc=lr_auroc
        )

        stats = [train_loss, *val_result, lr_auroc, lr_acc]
        writer.writerow([L - pbar.n] + [f"{s:.4f}" for s in stats])

        lr_models.append(lr_model)
        ccs_models.append(ccs_model)

    ccs_models.reverse()
    lr_models.reverse()

    path = elk_cache_dir() / args.name
    with open(path / "lr_models.pkl", "wb") as file:
        pickle.dump(lr_models, file)

    torch.save(ccs_models, path / "ccs_models.pt")


if __name__ == "__main__":
    args = get_training_parser().parse_args()
    train(args)
