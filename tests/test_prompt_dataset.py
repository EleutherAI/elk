from elk.extraction import Extract, PromptDataset
from elk.promptsource.templates import DatasetTemplates
import pytest


@pytest.mark.filterwarnings("ignore:Unable to find a decoding function")
def test_prompt_dataset_getitem_boolq():
    def test_prompt_dataset_getitem(cfg: Extract, split: str):
        prompt_ds = PromptDataset(cfg.prompts, rank=0, world_size=1, split=split)
        ds_name, _, config_name = cfg.prompts.dataset.partition(" ")

        prompter = DatasetTemplates(ds_name, config_name or None)
        assert len(prompt_ds) == cfg.prompts.max_examples[-1]
        for i in range(len(prompt_ds)):
            true_templates_ids = [
                template.id for template in prompter.templates.values()
            ]
            returned_prompts = prompt_ds[i]
            returned_templates_ids = [prompt.template.id for prompt in returned_prompts]

            # check for using the right example
            assert all(
                [
                    prompt_ds.active_split[i] == prompt.example
                    for prompt in returned_prompts
                ]
            )

            # check for using the same templates
            assert set(true_templates_ids) == set(returned_templates_ids)
            # check for them being in the same order
            assert true_templates_ids == returned_templates_ids

    # the case where the dataset has 2 classes
    # this dataset is small
    cfg = Extract.load_yaml("tests/distilgpt2_copa_cfg.yaml")
    test_prompt_dataset_getitem(cfg, "validation")

    # the case where the dataset has more than 2 classes
    # TODO: I'm not sure if we want to force people to download the whole dataset
    cfg = Extract.load_yaml("tests/distilgpt2_dbpedia_cfg.yaml")
    test_prompt_dataset_getitem(cfg, "test")
