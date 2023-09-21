from pathlib import Path

from elk import Extract
from elk.training.train import Elicit


def test_smoke_elicit_run_tiny_gpt2(tmp_path: Path):
    # we need about 5 mb of gpu memory to run this test
    model_path = "sshleifer/tiny-gpt2"
    dataset_name = "imdb"
    elicit = Elicit(
        data=Extract(
            model=model_path,
            datasets=(dataset_name,),
            max_examples=(10, 10),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        num_gpus=2,
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
    ]
    for file in expected_files:
        assert file in created_file_names
