from elk.extensions.recursive_ccs.rccs import RecursiveCCS
from elk.utils_evaluation.ccs import CCS, TrainParams
from elk.utils_evaluation.utils_evaluation import (
    get_hidden_states,
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
    hidden_states = get_hidden_states(
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

    print("Training CCS model...")
    x0, x1 = torch.from_numpy(features).to(args.device).chunk(2, dim=1)
    labels = torch.from_numpy(labels).to(args.device)

    rccs = RecursiveCCS(device=args.device)
    train_params = TrainParams(
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
    )

    save_path = args.trained_models_path / "rccs_model"
    n_probes = 2
    for i in range(n_probes):
        probe = rccs.fit_next_probe(data=(x0, x1), train_params=train_params)
        train_acc, train_loss = probe.score(data=(x0, x1), labels=labels)
        rccs.save_last_probe(save_path)
        print(f"Probe {i}: {train_acc=:.4f}, {train_loss=:.4f}")
    print("Done.")

    return rccs


if __name__ == "__main__":
    args = get_args(
        default_config_path=Path(__file__).parent.parent.parent / "default_config.json"
    )
    print(f"-------- args = {args} --------")

    rccs = train(args)
