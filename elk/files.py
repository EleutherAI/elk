"""Helper functions for dealing with files."""

import json
import os
import random
from pathlib import Path

import yaml
from simple_parsing import Serializable


def elk_reporter_dir() -> Path:
    """Return the directory where reporter checkpoints and logs are stored."""
    env_dir = os.environ.get("ELK_DIR", None)
    if env_dir is None:
        log_dir = Path.home() / "elk-reporters"
    else:
        log_dir = Path(env_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def memorably_named_dir(parent: Path):
    """Return a memorably-named cached directory of the form 'goofy-goodall'."""
    resource_dir = Path(__file__).parent / "resources"

    with open(resource_dir / "adjectives.json", "r") as f:
        adjectives = json.load(f)
    with open(resource_dir / "names.json", "r") as f:
        names = json.load(f)

    parent.mkdir(parents=True, exist_ok=True)
    sub_dir = "."

    while parent.joinpath(sub_dir).exists():
        adj = random.choice(adjectives)
        name = random.choice(names)
        sub_dir = f"{adj}-{name}"

    out_dir = parent / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config(cfg: Serializable, out_dir: Path):
    """Save the config to a file"""

    path = out_dir / "cfg.yaml"
    with open(path, "w") as f:
        cfg.dump_yaml(f)

    return path


def save_meta(dataset, out_dir: Path):
    """Save the meta data to a file"""

    meta = {
        "dataset_fingerprints": {
            split: dataset[split]._fingerprint for split in dataset.keys()
        }
    }
    path = out_dir / "metadata.yaml"
    with open(path, "w") as meta_f:
        yaml.dump(meta, meta_f)

    return path
