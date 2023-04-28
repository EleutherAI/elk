import hashlib
from typing import TYPE_CHECKING, Optional

import dill
from datasets import DatasetDict

from elk.extraction.dataset_name import DatasetDictWithName
from elk.files import elk_extract_cache_dir

if TYPE_CHECKING:
    from elk import Extract


def extract_cache_key(cfg: "Extract", ds_name: str) -> str:
    """Return a unique key for the extract cache."""
    cfg_str = str(cfg.to_dict())
    # hash it. note that the hash has to be deterministic
    cfg_hash = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()
    return f"{ds_name}-{cfg_hash}"


def load_extract_from_cache(
    ds_name: str, cache_key: str
) -> Optional[DatasetDictWithName]:
    # Use dill to load the DatasetDict
    path = elk_extract_cache_dir() / f"{cache_key}.dill"
    # check if the cache exists
    if not path.exists():
        return None
    with open(path, "rb") as f:
        dataset_dict = dill.load(f)
    assert isinstance(dataset_dict, DatasetDict)
    print(f"Loaded cached extract dataset from {path}")
    return DatasetDictWithName(name=ds_name, dataset=dataset_dict)


def maybe_load_extract_cache(
    cfg: "Extract", ds_name: str, disable_cache: bool
) -> DatasetDictWithName | None:
    if disable_cache:
        return None
    cache_key = extract_cache_key(cfg, ds_name)
    try:
        dataset_dict = load_extract_from_cache(ds_name=ds_name, cache_key=cache_key)
        return dataset_dict
    except Exception as e:
        print(f"Failed to load cached extract dataset {cache_key}: {e}")
        return None


def write_extract_to_cache(dataset_dict: DatasetDictWithName, cache_key: str) -> None:
    """Write a DatasetDictWithName to disk."""
    path = elk_extract_cache_dir() / f"{cache_key}.dill"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(dataset_dict.dataset, f)
