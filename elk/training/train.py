from ..files import elk_cache_dir
from .ccs import CCS
from .preprocessing import load_hidden_states, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
import csv
import numpy as np
import pickle
import random
import torch
import torch.distributed as dist


def train(args):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if dist.is_initialized() and not args.skip_baseline and rank == 0:
        print("Skipping LR baseline during distributed training.")

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

    train_hiddens, val_hiddens = normalize(
        train_hiddens, val_hiddens, args.normalization
    )
    if dist.is_initialized():
        world_size = dist.get_world_size()
        train_hiddens = train_hiddens.chunk(world_size)[rank]
        train_labels = train_labels.chunk(world_size)[rank]

        val_hiddens = val_hiddens.chunk(world_size)[rank]
        val_labels = val_labels.chunk(world_size)[rank]

    ccs_models = []
    lr_models = []
    L = train_hiddens.shape[1]

    # Do the last layer first- useful for debugging, maybe change later
    val_layers = list(val_hiddens.unbind(1))
    train_layers = list(train_hiddens.unbind(1))
    val_layers.reverse()
    train_layers.reverse()

    iterator = zip(train_layers, val_layers)
    pbar = None
    if rank == 0:
        pbar = tqdm(iterator, total=L, unit="layer")
        iterator = pbar

    statistics = []
    for i, (train_h, val_h) in enumerate(iterator):
        # Note: currently we're just upcasting to float32 so we don't have to deal with
        # grad scaling (which isn't supported for LBFGS), while the hidden states are
        # saved in float16 to save disk space. In the future we could try to use mixed
        # precision training in at least some cases.
        x0, x1 = train_h.to(args.device).float().chunk(2, dim=-1)
        val_x0, val_x1 = val_h.to(args.device).float().chunk(2, dim=-1)

        if pbar:
            pbar.set_description("Fitting CCS")
        ccs_model = CCS(
            in_features=x0.shape[-1], device=args.device, init=args.init, loss=args.loss
        )
        if args.label_frac:
            num_labels = round(args.label_frac * len(train_labels))
            labels = train_labels[:num_labels].to(args.device)
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
            val_labels.to(args.device),
        )
        if pbar:
            pbar.set_postfix(train_loss=train_loss, ccs_auroc=val_result.auroc)
        stats = [train_loss, *val_result]

        if not args.skip_baseline and not dist.is_initialized():
            # TODO: Once we implement cross-validation for CCS, we should benchmark
            # against LogisticRegressionCV here.

            train_labels_aug = torch.cat([train_labels, 1 - train_labels]).cpu()
            val_labels_aug = torch.cat([val_labels, 1 - val_labels]).cpu()

            if pbar:
                pbar.set_description("Fitting LR")
            lr_model = LogisticRegression(max_iter=10_000)
            lr_model.fit(torch.cat([x0, x1]).cpu(), train_labels_aug)

            lr_preds = lr_model.predict_proba(torch.cat([val_x0, val_x1]).cpu())[:, 1]
            lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
            lr_auroc = roc_auc_score(val_labels_aug, lr_preds)
            if pbar:
                pbar.set_postfix(
                    train_loss=train_loss, ccs_auroc=val_result.auroc, lr_auroc=lr_auroc
                )
            lr_models.append(lr_model)
            stats += [lr_auroc, lr_acc]

        statistics.append(stats)
        ccs_models.append(ccs_model)

    ccs_models.reverse()
    lr_models.reverse()

    path = elk_cache_dir() / args.name
    if rank == 0:
        cols = ["layer", "train_loss", "loss", "acc", "cal_acc", "auroc"]
        if not args.skip_baseline:
            cols += ["lr_auroc", "lr_acc"]

        with open(cache_dir / "eval.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(cols)

            for i, stats in enumerate(statistics):
                writer.writerow([L - i] + [f"{s:.4f}" for s in stats])

        torch.save(ccs_models, path / "ccs_models.pt")
        if lr_models:
            with open(path / "lr_models.pkl", "wb") as file:
                pickle.dump(lr_models, file)
