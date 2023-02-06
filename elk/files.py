from pathlib import Path
import json
import os
import pickle
import random


def elk_cache_dir() -> Path:
    """Return the path to the directory where files are stored."""
    env_dir = os.environ.get("ELK_CACHE_DIR", None)
    if env_dir is None:
        cache_dir = Path.home() / ".cache" / "elk"
    else:
        cache_dir = Path(env_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_memorable_cache_dir():
    """Return a memorably-named cached directory of the form 'goofy-goodall'."""
    resource_dir = Path(__file__).parent / "resources"

    with open(resource_dir / "adjectives.json", "r") as f:
        adjectives = json.load(f)
    with open(resource_dir / "names.json", "r") as f:
        names = json.load(f)

    file_name = "."
    root = elk_cache_dir()

    while root.joinpath(file_name).exists():
        adj = random.choice(adjectives)
        noun = random.choice(names)
        file_name = f"{adj}-{noun}"

    path = root / file_name
    path.mkdir(parents=True, exist_ok=True)
    return path
