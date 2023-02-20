"""Helper functions for dealing with files."""

from pathlib import Path
import os


def elk_cache_dir() -> Path:
    """Return the path to the directory where files are stored."""
    env_dir = os.environ.get("ELK_CACHE_DIR", None)
    if env_dir is None:
        cache_dir = Path.home() / ".cache" / "elk"
    else:
        cache_dir = Path(env_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
