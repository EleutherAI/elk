"""Helper functions for dealing with files."""

from .argparsers import get_saveable_args
from argparse import Namespace
from hashlib import md5
from pathlib import Path
import json
import os
import pickle
import random


def args_to_uuid(args: Namespace) -> str:
    """Generate a unique ID based on the input arguments."""

    identifying_args = get_saveable_args(args)

    return md5(pickle.dumps(identifying_args)).hexdigest()


def elk_cache_dir() -> Path:
    """Return the directory where extracted hidden states are cached."""
    env_dir = os.environ.get("ELK_CACHE_DIR", None)
    if env_dir is None:
        cache_dir = Path.home() / ".cache" / "elk"
    else:
        cache_dir = Path(env_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def elk_log_dir() -> Path:
    """Return the directory where logs and checkpoints are stored."""
    env_dir = os.environ.get("ELK_DIR", None)
    if env_dir is None:
        log_dir = elk_cache_dir() / "logs"
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

    sub_dir = "."
    while parent.joinpath(sub_dir).exists():
        adj = random.choice(adjectives)
        name = random.choice(names)
        sub_dir = f"{adj}-{name}"

    out_dir = parent / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
