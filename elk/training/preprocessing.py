from pathlib import Path
import torch


def normalize(data: torch.Tensor):
    variances, means = torch.var_mean(data, dim=0)
    return (data - means) / variances.add(1e-5).sqrt()


def load_hidden_states(path: Path):
    hiddens, labels = torch.load(path, map_location="cpu")

    # Normalize to zero mean and unit variance, then concatenate the
    # positive and negative examples together.
    normalized = normalize(hiddens.float()).flatten(start_dim=-2)
    return normalized, labels
