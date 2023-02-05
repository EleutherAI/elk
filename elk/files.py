from pathlib import Path
from random import Random
import json
import os
import pickle


def elk_cache_dir() -> Path:
    """Return the path to the directory where files are stored."""
    env_dir = os.environ.get("ELK_CACHE_DIR", None)
    if env_dir is None:
        cache_dir = Path.home() / ".cache" / "elk"
    else:
        cache_dir = Path(env_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_memorable_cache_dir(root: Path, seed=None, exist_ok: bool = False):
    """Return a memorably-named cached directory of the form 'goofy-goodall'."""
    resource_dir = Path(__file__).parent / "resources"

    with open(resource_dir / "adjectives.json", "r") as f:
        adjectives = json.load(f)
    with open(resource_dir / "nouns.json", "r") as f:
        nouns = json.load(f)

    rng = Random(x=pickle.dumps(seed) if seed is not None else None)
    root.mkdir(parents=True, exist_ok=True)

    adj = rng.choice(adjectives)
    noun = rng.choice(nouns)
    file_name = f"{adj}-{noun}"

    i = 1
    while not exist_ok and root.joinpath(file_name).exists():
        file_name = f"{adj}-{noun}-{i}"

    return root / file_name
