"""Preprocessing functions for training."""

from typing import Literal
import torch


def normalize(
    train_hiddens: torch.Tensor,
    val_hiddens: torch.Tensor,
    method: Literal["legacy", "elementwise", "meanonly"] = "legacy",
):
    """Normalize the hidden states.

    Normalize the hidden states with the specified method.
    The "legacy" method is the same as the original ELK implementation.
    The "elementwise" method normalizes each element.
    The "meanonly" method normalizes the mean.

    Args:
        train_hiddens: The hidden states for the training set.
        val_hiddens: The hidden states for the validation set.
        method: The normalization method to use.

    Returns:
        tuple containing the training and validation hidden states.
    """
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
