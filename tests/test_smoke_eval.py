from pathlib import Path
from typing import Sequence

import pandas as pd

from elk import Extract
from elk.evaluation import Eval
from elk.extraction import PromptConfig
from elk.extraction.dataset_name import extract_dataset_name_and_config
from elk.files import transfer_eval_directory
from elk.training import CcsReporterConfig, EigenReporterConfig
from elk.training.train import Elicit

EVAL_EXPECTED_FILES = [
    "cfg.yaml",
    "fingerprints.yaml",
    "eval.csv",
]


# TODO make into a pytest.fixture?
def setup_elicit(
    tmp_path: Path,
    dataset_name="imdb",
    model_path="sshleifer/tiny-gpt2",
    min_mem=10 * 1024 ** 2,
    is_ccs: bool = True,
) -> Elicit:
    """Setup elicit config for testing, execute elicit, and save output to tmp_path.
    Returns the elicit run configuration.
    """
    elicit = Elicit(
        data=Extract(
            model=model_path,
            prompts=PromptConfig(datasets=[dataset_name], max_examples=[10]),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        num_gpus=2,
        min_gpu_mem=min_mem,
        net=CcsReporterConfig() if is_ccs else EigenReporterConfig(),
        out_dir=tmp_path,
        disable_cache=True,
    )
    elicit.execute()
    return elicit


def check_contains_files(dir: Path, expected_files: list[str]):
    """Iterate through dir to assert all expected files exist."""
    files: list[Path] = list(dir.iterdir())
    created_file_names = {file.name for file in files}
    for file in expected_files:
        assert file in created_file_names


def eval_run(elicit: Elicit, transfer_datasets: Sequence[str] = []) -> int:
    """A single eval run; act and assert that expected files were created.
    Returns a reference time (in seconds) for file modification checking.
    """
    tmp_path = elicit.out_dir
    extract = elicit.data

    # record elicit modification time as reference.
    start_time_sec = (tmp_path / "eval.csv").stat().st_mtime

    if transfer_datasets:
        # update datasets to a different dataset
        extract.prompts.datasets = transfer_datasets

    eval = Eval(data=extract, source=tmp_path)
    eval.execute()
    return start_time_sec


def eval_assert_files_created(elicit: Elicit, transfer_datasets: Sequence[str] = []):
    tmp_path = elicit.out_dir
    eval_dir = transfer_eval_directory(source=tmp_path)
    assert eval_dir.exists(), f"transfer eval dir {eval_dir} does not exist"
    check_contains_files(eval_dir, EVAL_EXPECTED_FILES)
    # read "eval.csv" into a df
    df = pd.read_csv(eval_dir / "eval.csv")
    # get the "dataset" column
    dataset_col = df["dataset"]

    for tfr_dataset in transfer_datasets:
        # assert that the dataset column contains the transfer dataset
        ds_name, config_name = extract_dataset_name_and_config(tfr_dataset)
        assert ds_name in dataset_col.values


"""TESTS"""


def test_smoke_tfr_eval_run_tiny_gpt2_ccs(tmp_path: Path):
    elicit = setup_elicit(tmp_path)
    transfer_datasets = ["christykoh/imdb_pt"]
    eval_run(elicit, transfer_datasets=transfer_datasets)
    eval_assert_files_created(elicit, transfer_datasets=transfer_datasets)


def test_smoke_eval_run_tiny_gpt2_eigen(tmp_path: Path):
    elicit = setup_elicit(tmp_path, is_ccs=False)
    transfer_datasets = ["christykoh/imdb_pt"]
    eval_run(elicit, transfer_datasets=transfer_datasets)
    eval_assert_files_created(elicit, transfer_datasets=transfer_datasets)


def test_smoke_multi_eval_run_tiny_gpt2_ccs(tmp_path: Path):
    elicit = setup_elicit(tmp_path)
    transfer_datasets = ["christykoh/imdb_pt", "super_glue boolq"]
    eval_run(elicit, transfer_datasets=transfer_datasets)
    eval_assert_files_created(elicit, transfer_datasets=transfer_datasets)
