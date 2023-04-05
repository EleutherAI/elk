from itertools import cycle, islice
from typing import Literal

import pytest

from elk.extraction import PromptConfig, load_prompts
from elk.promptsource.templates import DatasetTemplates


@pytest.mark.filterwarnings("ignore:Unable to find a decoding function")
def test_load_prompts():
    def test_single_split(cfg: PromptConfig, split_type: Literal["train", "val"]):
        prompt_ds = load_prompts(
            *cfg.datasets,
            split_type=split_type,
        )
        prompters = []

        for ds in cfg.datasets:
            ds_name, _, config_name = ds.partition(" ")
            prompter = DatasetTemplates(ds_name, config_name or None)
            prompters.append(prompter)

        limit = cfg.max_examples[0 if split_type == "train" else 1]
        for prompter, record in zip(cycle(prompters), islice(prompt_ds, limit)):
            true_template_names = prompter.all_template_names
            returned_template_names = record["template_names"]

            # check for using the same templates
            assert set(true_template_names) == set(returned_template_names)
            # check for them being in the same order
            assert true_template_names == true_template_names

    # the case where the dataset has 2 classes
    # this dataset is small
    cfg = PromptConfig.load_yaml("tests/super_glue_prompts.yaml")
    test_single_split(cfg, "train")
    test_single_split(cfg, "val")

    # the case where the dataset has more than 2 classes
    cfg = PromptConfig.load_yaml("tests/dbpedia_prompts.yaml")
    test_single_split(cfg, "train")
    test_single_split(cfg, "val")
