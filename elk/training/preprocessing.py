from pathlib import Path
import logging
from typing import Literal
import torch
from datasets import load_from_disk


def normalize(
    train_hiddens: torch.Tensor,
    val_hiddens: torch.Tensor,
    method: Literal["legacy", "elementwise", "meanonly"] = "legacy",
):
    if method == "legacy":
        master = torch.cat([train_hiddens, val_hiddens], dim=0).float()
        means = master.mean(dim=0)

        train_hiddens -= means
        val_hiddens -= means

        scale = master.shape[-1] ** 0.5 / master.norm(dim=-1).mean()
        train_hiddens *= scale
        val_hiddens *= scale
    else:
        means = train_hiddens.float().mean(dim=0)
        train_hiddens -= means
        val_hiddens -= means

        if method == "elementwise":
            scale = 1 / train_hiddens.norm(dim=0, keepdim=True)
        elif method == "meanonly":
            scale = 1
        else:
            raise NotImplementedError(f"Scale method '{method}' is not supported.")

        train_hiddens *= scale
        val_hiddens *= scale

    return train_hiddens, val_hiddens


def load_hidden_states(path: Path):
    """Load the hidden states from a file."""
    # TODO: the training code should be updated to use all the information
    # in the dataset, not just the hidden states and labels. This is just
    # a temporary link to the old code.

    # load the huggingface dataset, then convert it to a torch tensor
    dataset = load_from_disk(path)
    labels = dataset["labels"]
    hiddens = torch.tensor(dataset["hiddens"], device="cpu")

    # Concatenate the positive and negative examples together.
    return hiddens.flatten(start_dim=-2), labels


def silence_datasets_messages():
    """Silence the annoying wall of logging messages and warnings."""

    def filter_fn(log_record):
        msg = log_record.getMessage()
        return (
            "Found cached dataset" not in msg
            and "Loading cached" not in msg
            and "Using custom data configuration" not in msg
        )

    handler = logging.StreamHandler()
    handler.addFilter(filter_fn)
    logging.getLogger("datasets").addHandler(handler)
