import json
from elk.eval.parser import get_parser


def test_args_no_underscores():
    """
    Assert that there are no underscores in the CLI parameter names. We wish
    to enforce this style since the GNU style guide recommends against it
    and terminal UX issues can cause underscores to be dropped. Dashes
    should be used instead. EX: batch-size=X instead of batch_size=X
    """

    datasets = None
    models = None
    prefix = None
    with open("elk/resources/default_config.json", "r", encoding="utf-8") as f:
        default_config = json.load(f)
        datasets = default_config["datasets"]
        models = default_config["models"]
        prefix = default_config["prefix"]

    parser = get_parser(datasets, models, prefix)
    for cli_flag in parser._actions:
        for flag_string in cli_flag.option_strings:
            assert "_" not in flag_string
