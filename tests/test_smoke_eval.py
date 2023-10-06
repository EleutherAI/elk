from pathlib import Path

import pandas as pd
import pytest

from elk import Extract
from elk.evaluation import Eval
from elk.training.train import Elicit

EVAL_EXPECTED_FILES = [
    "cfg.yaml",
    "fingerprints.yaml",
    "lr_eval.csv",
]


# TODO make into a pytest.fixture?
def setup_elicit(
    tmp_path: Path,
    dataset_name="imdb",
    model_path="sshleifer/tiny-gpt2",
) -> Elicit:
    """Setup elicit config for testing, execute elicit, and save output to tmp_path.
    Returns the elicit run configuration.
    """
    elicit = Elicit(
        data=Extract(
            model=model_path,
            datasets=(dataset_name,),
            max_examples=(10, 10),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        num_gpus=1,
        out_dir=tmp_path,
        min_gpu_mem=5_000_000,
    )
    elicit.execute()
    return elicit


def check_contains_files(dir: Path, expected_files: list[str]):
    """Iterate through dir to assert all expected files exist."""
    files: list[Path] = list(dir.iterdir())
    created_file_names = {file.name for file in files}
    for file in expected_files:
        assert file in created_file_names


def eval_run(elicit: Elicit, transfer_datasets: tuple[str, ...] = ()) -> float:
    """A single eval run; act and assert that expected files were created.
    Returns a reference time (in seconds) for file modification checking.
    """
    tmp_path = elicit.out_dir
    extract = elicit.data
    assert tmp_path is not None

    # record elicit modification time as reference.
    start_time_sec = (tmp_path / "lr_eval.csv").stat().st_mtime

    if transfer_datasets:
        # update datasets to a different dataset
        extract.datasets = transfer_datasets

    eval = Eval(data=extract, source=tmp_path)
    eval.execute()
    return start_time_sec


def eval_assert_files_created(elicit: Elicit, transfer_datasets: tuple[str, ...] = ()):
    tmp_path = elicit.out_dir
    assert tmp_path is not None

    eval_dir = tmp_path / "transfer" / "+".join(transfer_datasets)
    assert eval_dir.exists(), f"transfer eval dir {eval_dir} does not exist"
    check_contains_files(eval_dir, EVAL_EXPECTED_FILES)
    # read "eval.csv" into a df
    df = pd.read_csv(eval_dir / "lr_eval.csv")
    # get the "dataset" column
    dataset_col = df["dataset"]

    for tfr_dataset in transfer_datasets:
        # assert that the dataset column contains the transfer dataset
        assert tfr_dataset in dataset_col.values


"""TESTS"""


@pytest.mark.gpu
def test_smoke_eval_run_tiny_gpt2(tmp_path: Path):
    elicit = setup_elicit(tmp_path)
    transfer_datasets = ("christykoh/imdb_pt",)
    eval_run(elicit, transfer_datasets=transfer_datasets)
    eval_assert_files_created(elicit, transfer_datasets=transfer_datasets)
