import pytest
import os


# test that running both elicit and extract works
# uses os.system to run the commands similar to how they would be run in the terminal
# tests that readme example works
@pytest.mark.cpu
def test_elk_run():
    # run elicit - check that the command runs without error
    assert (
        os.waitstatus_to_exitcode(
            os.system(
                "NO_CUDA=1 elk elicit"
                + " microsoft/deberta-v2-xxlarge-mnli imdb --max_examples 10"
            )
        )
        == 0
    ), "elicit command failed"
    # run extract - check that the command runs without error
    assert (
        os.waitstatus_to_exitcode(
            os.system(
                "NO_CUDA=1 elk extract"
                + " microsoft/deberta-v2-xxlarge-mnli imdb --max_examples 10"
            )
        )
        == 0
    ), "extract command failed"
