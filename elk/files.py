from argparse import Namespace
from hashlib import md5
from pathlib import Path
import os
import pickle


def args_to_uuid(args: Namespace) -> str:
    """Return a unique identifier for the given CLI args."""

    identifying_args = vars(args).copy()
    del identifying_args["device"]  # Device shouldn't affect the output

    return md5(pickle.dumps(identifying_args)).hexdigest()


def elk_cache_dir() -> Path:
    """Return the path to the directory where files are stored."""
    env_dir = os.environ.get("ELK_CACHE_DIR", None)
    if env_dir is None:
        cache_dir = Path.home() / ".cache" / "elk"
    else:
        cache_dir = Path(env_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
