from pathlib import Path

from elk import ExtractionConfig
from elk.extraction import extract, PromptConfig
from elk.training.train import train, RunConfig


def test_smoke_elicit_run(tmp_path: Path):
    # We'll use the tiny DeBERTa model for this test
    model_path = "hf-internal-testing/tiny-deberta"
    # todo: support tiny-imdb. But somnehow we need to convince promptsource
    dataset_name = "imdb"
    config = RunConfig(
        data=ExtractionConfig(
            model=model_path,
            prompts=PromptConfig(dataset=dataset_name, max_examples=[10]),
            layers=(1,),
        ),
    )
    dataset = train(config)
    dataset.save_to_disk(tmp_path)
    # get the files in the tmp_path
    files = list(tmp_path.iterdir())
