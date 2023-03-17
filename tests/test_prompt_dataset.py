from elk.extraction import extract, ExtractionConfig, PromptDataset
from elk.promptsource.templates import DatasetTemplates


def test_prompt_dataset_getitem():
    cfg = ExtractionConfig.load_yaml("tests/distilgpt2_boolq_cfg.yaml")
    prompt_ds = PromptDataset(cfg.prompts, rank=0, world_size=1, split="validation")

    ds_name, config_name = cfg.prompts.dataset.split(" ")
    prompter = DatasetTemplates(ds_name, config_name or None)
    assert len(prompt_ds) == 2
    for i in range(len(prompt_ds)):
        true_prompts = [
            "\n".join(template.apply(prompt_ds.active_split[i]))
            for template in prompter.templates.values()
        ]
        # TODO in future use
        # true_prompts = [
        #   apply_template(template, boolq[i])
        #   for template_name, template in prompter.templates.items()
        # ]
        returned_prompts = prompt_ds[i]
        prompts = [
            ret_prompt.to_string(answer_idx=ret_prompt.label)
            for ret_prompt in returned_prompts
        ]

        # this checks if the prompts are the same, AND the labels are the same
        assert set(true_prompts) == set(prompts)
        # TODO: in future: assert true_prompts == prompts
