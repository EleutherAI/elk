import pytest
from elk.utils_evaluation.parser import get_args

# @pytest.marks.cpu
def test_args_no_lowercase():
    arguments = get_args("elk/default_config.json")


