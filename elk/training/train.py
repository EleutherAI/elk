"""Main training loop."""

from ..files import elk_cache_dir
from .preprocessing import load_hidden_states, normalize
from .reporter import Reporter
from argparse import Namespace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
import csv
import numpy as np
import pickle
import random
import torch
import torch.multiprocessing as mp


def train_task(input_q: mp.Queue, out_q: mp.Queue, args: Namespace, rank: int = 0):
    """Worker function for training reporters in parallel."""

    while not input_q.empty():
        args.device = torch.device(f"cuda:{rank}")

        i, *data = input_q.get()
        out_q.put((i, *train_reporter(args, i, *data)))


def train_reporter(
    args: Namespace,
    layer_index: int,
    train_h: torch.Tensor,
    val_h: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
):
    """Train a single reporter on a single layer."""

    # Note: currently we're just upcasting to float32 so we don't have to deal with
    # grad scaling (which isn't supported for LBFGS), while the hidden states are
    # saved in float16 to save disk space. In the future we could try to use mixed
    # precision training in at least some cases.
    x0, x1 = train_h.to(args.device).float().chunk(2, dim=-1)
    val_x0, val_x1 = val_h.to(args.device).float().chunk(2, dim=-1)

    train_labels_aug = torch.cat([train_labels, 1 - train_labels])
    val_labels_aug = torch.cat([val_labels, 1 - val_labels])

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

    output_dir = elk_cache_dir() / args.name
    lr_dir = output_dir / "lr_models"
    reporter_dir = output_dir / "reporters"

    lr_dir.mkdir(parents=True, exist_ok=True)
    reporter_dir.mkdir(parents=True, exist_ok=True)
    stats = [train_loss, *val_result]

    if not args.skip_baseline and not args.num_devices > 1:
        # TODO: Once we implement cross-validation for CCS, we should benchmark
        # against LogisticRegressionCV here.
        lr_model = LogisticRegression(max_iter=10_000)
        lr_model.fit(torch.cat([x0, x1]).cpu(), train_labels_aug)

        lr_preds = lr_model.predict_proba(torch.cat([val_x0, val_x1]).cpu())[:, 1]
        lr_acc = accuracy_score(val_labels_aug, lr_preds > 0.5)
        lr_auroc = roc_auc_score(val_labels_aug, lr_preds)

        stats += [lr_auroc, lr_acc]
        with open(lr_dir / f"layer_{layer_index}.pkl", "wb") as file:
            pickle.dump(lr_model, file)

    with open(reporter_dir / f"layer_{layer_index}.pt", "wb") as file:
        torch.save(reporter, file)

    return stats


def train(args):
    # This is needed to use multiprocessing with CUDA.
    mp.set_start_method("spawn")

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

    train_layers = list(train_hiddens.share_memory_().unbind(1))
    val_layers = list(val_hiddens.share_memory_().unbind(1))
    iterator = zip(train_layers, val_layers)

    # Collect the results and update the progress bar.
    with open(cache_dir / "eval.csv", "w") as f:
        cols = ["layer", "train_loss", "loss", "acc", "cal_acc", "auroc"]
        if not args.skip_baseline:
            cols += ["lr_auroc", "lr_acc"]

        writer = csv.writer(f)
        writer.writerow(cols)

        # Don't bother with multiprocessing if we're only using one device.
        if args.num_devices == 1:
            pbar = tqdm(iterator, total=L, unit="layer")
            for i, (train_h, val_h) in enumerate(pbar):
                train_reporter(
                    args,
                    i,
                    train_h,
                    val_h,
                    train_labels,
                    val_labels,
                )
        else:
            # Use one queue to distribute the work to the workers, and another to
            # collect the results.
            data_queue = mp.Queue()
            result_queue = mp.Queue()
            for i, (train_h, val_h) in enumerate(iterator):
                data_queue.put((i, train_h, val_h, train_labels, val_labels))

            # Apparently the .put() command is non-blocking, so we need to wait
            # until the queue is filled before starting the workers.
            while data_queue.empty():
                pass

            # Start the workers.
            workers = []
            for i in range(args.num_devices):
                worker = mp.Process(
                    target=train_task,
                    args=(data_queue, result_queue, args),
                    kwargs=dict(rank=i),
                )
                worker.start()
                workers.append(worker)

            pbar = tqdm(total=L, unit="layer")
            while not data_queue.empty() or not result_queue.empty():
                i, *stats = result_queue.get()
                pbar.update()
                writer.writerow([L - i] + [f"{s:.4f}" for s in stats])

            # Workers should be done by now.
            pbar.close()
            for worker in workers:
                worker.join()
