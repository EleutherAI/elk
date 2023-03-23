from pathlib import Path

from elk import ExtractionConfig
from elk.extraction import extract, PromptConfig
from elk.training import CcsReporterConfig
from elk.training.train import train, RunConfig


def test_smoke_elicit_run(tmp_path: Path):
    # We'll use the tiny gpt2 model for this test
    model_path = "sshleifer/tiny-gpt2"
    # todo: support tiny-imdb. But somnehow we need to convince promptsource
    dataset_name = "imdb"
    config = RunConfig(
        data=ExtractionConfig(
            model=model_path,
            prompts=PromptConfig(dataset=dataset_name, max_examples=[10]),
            layers=(1,),
        ),
        net=CcsReporterConfig(),
    )
    dataset = train(config, tmp_path)
    # get the files in the tmp_path
    files: Path = list(tmp_path.iterdir())
    created_file_names = {file.name for file in files}
    expected_files = ["cfg.yaml", "metadata.yaml", "lr_models", "reporters", "eval.csv"]
    for file in expected_files:
        assert file in created_file_names
