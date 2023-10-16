from pathlib import Path

import pytest

from elk.plotting.visualize import SweepVisualization


@pytest.fixture
def setup_fs(fs):
    test_dir = "/sweep1"
    fs.create_dir(test_dir)
    fs.create_dir(f"{test_dir}/huggyllama/llama-13b/imdb")
    fs.create_file(f"{test_dir}/huggyllama/llama-13b/imdb/eval.csv")
    fs.create_dir(f"{test_dir}/huggyllama/llama-12b/news")
    fs.create_file(f"{test_dir}/huggyllama/llama-12b/news/eval.csv")
    fs.create_file(f"{test_dir}/gpt2-medium/imdb/eval.csv")

    return Path(test_dir)


# def test_get_model_paths(setup_fs):
#     test_dir = setup_fs
#     result = SweepVisualization._get_model_paths(test_dir)
#
#     root = Path(test_dir)
#     for path in root.rglob("*"):
#         print(path)
#     assert len(result) == 3
#     assert any([p.name == "llama-13b" for p in result])
#     assert any([p.name == "llama-12b" for p in result])
#     assert any([p.name == "gpt2-medium" for p in result])
