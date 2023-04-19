import pytest


@pytest.mark.gpu
def test_gpu_example():
    """Will only run if the `gpu` mark is specified
    This is just an example test to show how to use the `gpu` mark
    We'll need to implement a GPU runner in the CI for actual GPU tests
    GPU tests can be run with `pytest -m gpu`"""
    assert True
