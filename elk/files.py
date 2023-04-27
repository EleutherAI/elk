"""Helper functions for dealing with files."""

import json
import os
import random
from pathlib import Path


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


def transfer_eval_directory(source: Path) -> Path:
    """Return the directory where transfer evals are stored."""
    return elk_reporter_dir() / source / "transfer_eval"
