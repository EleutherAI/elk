import unittest
from pathlib import Path

from pyfakefs.fake_filesystem_unittest import TestCase

from elk.plotting.visualize import SweepVisualization


class TestGetModelPaths(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

        # Create a test directory structure
        self.test_dir = "/sweep1"
        self.fs.create_dir(self.test_dir)
        self.fs.create_dir(f"{self.test_dir}/huggyllama/llama-13b/imdb")
        self.fs.create_file(f"{self.test_dir}/huggyllama/llama-13b/imdb/eval.csv")
        self.fs.create_dir(f"{self.test_dir}/huggyllama/llama-12b/news")
        self.fs.create_file(f"{self.test_dir}/huggyllama/llama-12b/news/eval.csv")
        self.fs.create_file(f"{self.test_dir}/gpt2-medium/imdb/eval.csv")

    def test_get_model_paths(self):
        result = SweepVisualization._get_model_paths(Path(self.test_dir))
        self.assertEqual(len(result), 3)
        self.assertTrue(any([p.name == "llama-13b" for p in result]))
        self.assertTrue(any([p.name == "llama-12b" for p in result]))
        self.assertTrue(any([p.name == "gpt2-medium" for p in result]))


if __name__ == "__main__":
    unittest.main()
