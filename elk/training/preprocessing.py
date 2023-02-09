from pathlib import Path
import logging
import torch


def normalize(data: torch.Tensor):
    variances, means = torch.var_mean(data, dim=0)
    return (data - means) / variances.add(1e-5).sqrt()


def load_hidden_states(path: Path):
    hiddens, labels = torch.load(path, map_location="cpu")

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
