"""Main training loop. Invokes the reporter."""

from ..files import elk_cache_dir
from .preprocessing import load_hidden_states, normalize
from .reporter import Reporter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
import csv
import numpy as np
import pickle
import random
import torch
import torch.multiprocessing as mp

def train_task(queue, args, reporters, lr_models, statistics):
    while not queue.empty():
        i, train_h, val_h, train_labels, val_labels = queue.get()
        args.device = torch.device(f"cuda:{mp.current_process()._identity[0] - 1}")
        train_probe(args, i, train_h, val_h, train_labels, val_labels, reporters, lr_models, statistics)
    return True

def train_probe(args, index, train_h, val_h, train_labels, val_labels, reporters, lr_models, statistics, pbar=None):
    # Note: currently we're just upcasting to float32 so we don't have to deal with
    #     # grad scaling (which isn't supported for LBFGS), while the hidden states are
    #     # saved in float16 to save disk space. In the future we could try to use mixed
    #     # precision training in at least some cases.
    x0, x1 = train_h.to(args.device).float().chunk(2, dim=-1)
    val_x0, val_x1 = val_h.to(args.device).float().chunk(2, dim=-1)

    train_labels_aug = torch.cat([train_labels, 1 - train_labels])
    val_labels_aug = torch.cat([val_labels, 1 - val_labels])
    if pbar:
        pbar.set_description("Fitting CCS")

    reporter = Reporter(
        in_features=x0.shape[-1],
        device=args.device,
        init=args.init,
        loss=args.loss,
        supervised_weight=args.supervised_weight,
    )
    if args.label_frac:
        num_labels = round(args.label_frac * len(train_labels))
        labels = train_labels[:num_labels].to(args.device)
    else:
        labels = None

    train_loss = reporter.fit(
        contrast_pair=(x0, x1),
        labels=labels,
        num_tries=args.num_tries,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
    )
    val_result = reporter.score(
        (val_x0, val_x1),
        val_labels.to(args.device),
    )
    if pbar:
        pbar.set_postfix(train_loss=train_loss, ccs_auroc=val_result.auroc)
    stats = [train_loss, *val_result]

    if not args.skip_baseline and not args.n_devices > 1:
        if pbar:
            pbar.set_description("Fitting LR")

        # TODO: Once we implement cross-validation for CCS, we should benchmark
        # against LogisticRegressionCV here.
        lr_model = LogisticRegression(max_iter=10_000)
        lr_model.fit(torch.cat([x0, x1]).cpu(), train_labels_aug)

        lr_preds = lr_model.predict_proba(torch.cat([val_x0, val_x1]).cpu())[:, 1]
        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)
        if pbar:
            pbar.set_postfix(
                train_loss=train_loss, ccs_auroc=val_result.auroc, lr_auroc=lr_auroc
            )

        lr_models.append((index, lr_model))
        stats += [lr_auroc, lr_acc]

    statistics.append((index, stats))
    reporters.append((index, reporter))

def train(args):
    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the training hidden states.
    cache_dir = elk_cache_dir() / args.name
    train_hiddens, train_labels = load_hidden_states(
        path=cache_dir / "train_hiddens.pt"
    )

    # load the validation hidden states.
    val_hiddens, val_labels = load_hidden_states(
        path=cache_dir / "validation_hiddens.pt"
    )

    # Ensure that the states are valid.
    assert len(set(train_labels)) > 1
    assert len(set(val_labels)) > 1

    # Normalize the hidden states with the specified method.
    train_hiddens, val_hiddens = normalize(
        train_hiddens, val_hiddens, args.normalization
    )

    L = train_hiddens.shape[1]

    # Do the last layer first- useful for debugging, maybe change later
    val_layers = list(val_hiddens.unbind(1))
    train_layers = list(train_hiddens.unbind(1))

    iterator = zip(train_layers, val_layers)

    reporters = []
    lr_models = []
    statistics = []

    if args.num_devices == 1:
        pbar = tqdm(iterator, total=L, unit="layer")
        iterator = pbar
        for i, (train_h, val_h) in enumerate(iterator):
            train_probe(args, i, train_h, val_h, train_labels, val_labels, reporters, lr_models, statistics, pbar)

    if args.num_devices > 1:
        queue = mp.Queue()
        for i, (train_h, val_h) in enumerate(iterator):
            queue.put((i, train_h, val_h, train_labels, val_labels))
        workers = []
        for _ in range(args.num_devices):
            worker = mp.Process(target=train_task, args=(queue, args, reporters, lr_models, statistics))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()

    reporters.sort()
    lr_models.sort()
    statistics.sort()
    reporters = list(map(lambda x: x[1], reporters))
    lr_models = list(map(lambda x: x[1], lr_models))
    statistics = list(map(lambda x: x[1], statistics))

    path = elk_cache_dir() / args.name
    cols = ["layer", "train_loss", "loss", "acc", "cal_acc", "auroc"]
    if not args.skip_baseline:
        cols += ["lr_auroc", "lr_acc"]

    with open(cache_dir / "eval.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

        for i, stats in enumerate(statistics):
            writer.writerow([L - i] + [f"{s:.4f}" for s in stats])

    torch.save(reporters, path / "reporters.pt")
    if lr_models:
        with open(path / "lr_models.pkl", "wb") as file:
            pickle.dump(lr_models, file)
