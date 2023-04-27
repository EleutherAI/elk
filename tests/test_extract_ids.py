from pathlib import Path

from transformers import AutoModelForCausalLM

from elk import Extract
from elk.extraction import PromptConfig
from elk.extraction.extraction import extract_input_ids
from elk.training import CcsReporterConfig
from elk.training.train import Elicit


def test_extract(tmp_path: Path):
    # we need about 5 mb of gpu memory to run this test
    model_path, min_mem = "sshleifer/tiny-gpt2", 10 * 1024**2
    dataset_name = "imdb"
    extract = Extract(
            model=model_path,
            prompts=PromptConfig(datasets=[dataset_name], max_examples=[10]),
            # run on all layers, tiny-gpt only has 2 layers
        )
    model = AutoModelForCausalLM.from_pretrained(model_path)

    result = extract_input_ids(cfg=extract, model=model, split_type="train")
    print("ok")
