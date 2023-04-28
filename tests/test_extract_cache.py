from pathlib import Path

from elk import Extract
from elk.extraction import PromptConfig, extract
from elk.extraction.caching import extract_cache_key, load_extract_from_cache
from elk.extraction.dataset_name import DatasetDictWithName
from elk.files import elk_extract_cache_dir
from elk.inference_server.fsdp_options import FSDPOptions


def test_cache_key():
    extract_cfg = Extract(
        model="sshleifer/tiny-gpt2",
        prompts=PromptConfig(datasets=["imdb"], max_examples=[10]),
    )
    # Use a different number of max_examples, the cache should not be used
    extract_cfg_new = Extract(
        model="sshleifer/tiny-gpt2",
        prompts=PromptConfig(datasets=["imdb"], max_examples=[100]),
    )
    assert extract_cache_key(cfg=extract_cfg, ds_name="imdb") != extract_cache_key(
        cfg=extract_cfg_new, ds_name="imdb"
    )


def test_extract_cache(tmp_path: Path):
    extract_cfg = Extract(
        model="sshleifer/tiny-gpt2",
        prompts=PromptConfig(datasets=["imdb"], max_examples=[10]),
    )

    extracted: DatasetDictWithName = extract(
        cfg=extract_cfg, fsdp=FSDPOptions(), disable_cache=False
    )

    key = extract_cache_key(cfg=extract_cfg, ds_name="imdb")
    path = elk_extract_cache_dir() / f"{key}.dill"
    assert path.exists()

    # load the cache
    loaded_from_cache: DatasetDictWithName | None = load_extract_from_cache(
        ds_name="imdb", cache_key=key
    )
    assert loaded_from_cache is not None
    extracted_dataset = loaded_from_cache.dataset
    loaded_dataset = extracted.dataset
    assert len(extracted_dataset) == len(loaded_dataset)
