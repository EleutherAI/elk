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


def normalize(data: torch.Tensor):
    variances, means = torch.var_mean(data, dim=0)
    return (data - means) / variances.sqrt()


def load_hidden_states(path: Path, reduce: str):
    hiddens, labels = torch.load(path, map_location="cpu")

    normalized = normalize(hiddens.float())
    normalized = reduce_paired_states(hiddens, reduce)

    return normalized, labels
