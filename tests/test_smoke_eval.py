from pathlib import Path

from elk import Extract
from elk.evaluation import Eval
from elk.extraction import PromptConfig
from elk.training import CcsReporterConfig, EigenReporterConfig
from elk.training.train import Elicit
from tests.utils import *

ELICIT_EXPECTED_FILES = [
    "cfg.yaml",
    "fingerprints.yaml",
    "lr_models",
    "reporters",
    "eval.csv",
]

EVAL_EXPECTED_FILES = [
    "cfg.yaml",
    "metadata.yaml",
    "eval.csv",
]


# TODO make into a pytest.fixture?
def setup_elicit(
    tmp_path: Path,
    dataset_name="imdb",
    model_path="sshleifer/tiny-gpt2",
    min_mem=10 * 1024**2,
    is_ccs: bool = True,
):
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
    )
    elicit.execute()
    check_contains_files(tmp_path, ELICIT_EXPECTED_FILES)
    return elicit


def check_contains_files(dir: Path, expected_files: list[str]):
    """Iterate through dir to assert all expected files exist."""
    files: list[Path] = list(dir.iterdir())
    created_file_names = {file.name for file in files}
    for file in expected_files:
        assert file in created_file_names


def eval_run(elicit: Elicit, tfr_datasets: list[str] = None):
    """A single eval run."""

    # record elicit modification time as reference.
    starttime = (tmp_path / "eval.csv").stat().st_mtime

    eval_only = tfr_datasets is not None

    if tfr_datasets:
        # update datasets to a different dataset
        elicit.data.prompts.datasets = tfr_datasets

    eval = Eval(data=elicit, source=tmp_path)
    eval.execute()

    if eval_only:
        assert (tmp_path / "eval.csv").stat().st_mtime > starttime
    else:
        # check transfer eval dir contents
        transfer_dir = tmp_path / "transfer_evals"
        assert transfer_dir.exists()
        eval_dir = transfer_dir / tfr_dataset
        assert eval_dir.exists()
        check_contains_files(eval_dir, EVAL_EXPECTED_FILES)


def test_smoke_eval_run_tiny_gpt2_ccs(tmp_path: Path):
    elicit = setup_elicit(tmp_path, is_ccs=True)
    Eval(data=elicit, source=tmp_path)
    eval_run(elicit)


def test_smoke_tfr_eval_run_tiny_gpt2_ccs(tmp_path: Path):
    elicit = setup_elicit(
        tmp_path,
        model_path="sshleifer/tiny-gpt2",
        min_mem=10 * 1024**2,
        dataset_name="imdb",
    )
    eval_run(elicit, tfr_datasets=["christykoh/imdb_pt"])


def test_smoke_multi_eval_run_tiny_gpt2_ccs():
    elicit = setup_elicit(
        tmp_path,
        model_path="sshleifer/tiny-gpt2",
        min_mem=10 * 1024**2,
        dataset_name="imdb",
    )
    eval_run(elicit, tfr_datasets=["super_glue boolq", "ag_news"])


def test_smoke_eval_run_tiny_gpt2_eigen(tmp_path: Path):
    elicit = setup_elicit(
        tmp_path,
        model_path="sshleifer/tiny-gpt2",
        min_mem=10 * 1024**2,
        dataset_name="imdb",
        is_ccs=False,
    )
    eval_run(elicit)
