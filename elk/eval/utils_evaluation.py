from pathlib import Path
import json
import torch

default_config_path = Path(__file__).parent.parent / "default_config.json"

with open(default_config_path, "r") as f:
    default_config = json.load(f)
datasets = default_config["datasets"]
model_shortcuts = default_config["model_shortcuts"]
prefix = default_config["prefix"]


def reduce_paired_states(hidden_states: torch.Tensor, mode: str):
    """Reduce pairs of hidden states into single vectors"""
    if mode.isdigit():
        return hidden_states[..., int(mode), :]
    elif mode == "minus":
        return hidden_states[..., 0, :] - hidden_states[..., 1, :]
    elif mode == "concat":
        return hidden_states.flatten(start_dim=-2)

    raise NotImplementedError("This mode is not supported.")


def normalize(data: torch.Tensor, scale=True, demean=True):
    # demean the array and rescale each data point
    data = data - data.mean(dim=0) if demean else data
    if not scale:
        return data

    return data / data.norm(dim=1).mean() * data.shape[1] ** 0.5


def load_hidden_states(
    path: Path,
    reduce: str,
    scale=True,
    demean=True,
):
    hiddens, labels = torch.load(path, map_location="cpu")

    normalized = normalize(hiddens, scale, demean)
    normalized = reduce_paired_states(hiddens, reduce)

    return normalized, labels
