import pytest
from elk.argparsers import get_extraction_parser


@pytest.mark.cpu
def test_args_no_underscores():
    """
    Assert that there are no underscores in the CLI parameter names. We wish
    to enforce this style since the GNU style guide recommends against it
    and terminal UX issues can cause underscores to be dropped. Dashes
    should be used instead. EX: batch-size=X instead of batch_size=X
    """

    parser = get_extraction_parser()
    for cli_flag in parser._actions:
        for flag_string in cli_flag.option_strings:
            assert "_" not in flag_string
