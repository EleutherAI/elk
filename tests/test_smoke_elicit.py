from pathlib import Path

from elk import Extract
from elk.training import CcsConfig, EigenFitterConfig
from elk.training.train import Elicit


def test_smoke_elicit_run_tiny_gpt2_ccs(tmp_path: Path):
    # we need about 5 mb of gpu memory to run this test
    model_path, min_mem = "sshleifer/tiny-gpt2", 10 * 1024**2
    dataset_name = "imdb"
    elicit = Elicit(
        data=Extract(
            model=model_path,
            datasets=(dataset_name,),
            max_examples=(10, 10),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        num_gpus=2,
        min_gpu_mem=min_mem,
        net=CcsConfig(),
        out_dir=tmp_path,
    )
    elicit.execute()
    # get the files in the tmp_path
    files: list[Path] = list(tmp_path.iterdir())
    created_file_names = {file.name for file in files}
    expected_files = [
        "cfg.yaml",
        "fingerprints.yaml",
        "lr_models",
        "reporters",
        "eval.csv",
        "layer_ensembling.csv",
    ]
    for file in expected_files:
        assert file in created_file_names


def test_smoke_elicit_run_tiny_gpt2_eigen(tmp_path: Path):
    # we need about 5 mb of gpu memory to run this test
    model_path, min_mem = "sshleifer/tiny-gpt2", 10 * 1024**2
    dataset_name = "imdb"
    elicit = Elicit(
        data=Extract(
            model=model_path,
            datasets=(dataset_name,),
            max_examples=(10, 10),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        num_gpus=2,
        min_gpu_mem=min_mem,
        net=EigenFitterConfig(),
        out_dir=tmp_path,
    )
    elicit.execute()
    # get the files in the tmp_path
    files: list[Path] = list(tmp_path.iterdir())
    created_file_names = {file.name for file in files}
    expected_files = [
        "cfg.yaml",
        "fingerprints.yaml",
        "lr_models",
        "reporters",
        "eval.csv",
        "layer_ensembling.csv",
    ]
    for file in expected_files:
        assert file in created_file_names
