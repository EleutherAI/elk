from itertools import islice
from typing import Literal

import pytest

from elk.extraction import Extract, load_prompts
from elk.promptsource.templates import DatasetTemplates


@pytest.mark.filterwarnings("ignore:Unable to find a decoding function")
def test_load_prompts():
    def test_single_split(cfg: Extract, split_type: Literal["train", "val"]):
        for cfg in cfg.explode():
            ds_string = cfg.datasets[0]
            prompt_ds = load_prompts(ds_string, split_type=split_type)

            ds_name, _, config_name = ds_string.partition(":")
            prompter = DatasetTemplates(ds_name, config_name or None)
            prompter.drop_non_mc_templates()

            limit = cfg.max_examples[0 if split_type == "train" else 1]
            for record in islice(prompt_ds, limit):
                true_template_names = prompter.all_template_names
                returned_template_names = record["template_names"]

                # check for using the same templates
                assert set(true_template_names) == set(returned_template_names)
                # check for them being in the same order
                assert true_template_names == true_template_names

    # the case where the dataset has 2 classes
    # this dataset is small
    cfg = Extract.load_yaml("tests/super_glue_prompts.yaml")
    test_single_split(cfg, "train")
    test_single_split(cfg, "val")

    # the case where the dataset has more than 2 classes
    cfg = Extract.load_yaml("tests/dbpedia_prompts.yaml")
    test_single_split(cfg, "train")
    test_single_split(cfg, "val")
