from pathlib import Path

from elk import ExtractionConfig
from elk.extraction import PromptConfig
from elk.training import CcsReporterConfig, EigenReporterConfig
from elk.training.train import train, RunConfig

"""
TODO: These tests should work with deberta
but you'll need to make deberta fp32 instead of fp16
because pytorch cpu doesn't support fp16
"""


def test_smoke_elicit_run_tiny_gpt2_ccs(tmp_path: Path):
    model_path = "sshleifer/tiny-gpt2"
    dataset_name = "imdb"
    config = RunConfig(
        data=ExtractionConfig(
            model=model_path,
            prompts=PromptConfig(dataset=dataset_name, max_examples=[10]),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        net=CcsReporterConfig(),
    )
    train(config, tmp_path)
    # get the files in the tmp_path
    files: Path = list(tmp_path.iterdir())
    created_file_names = {file.name for file in files}
    expected_files = ["cfg.yaml", "metadata.yaml", "lr_models", "reporters", "eval.csv"]
    for file in expected_files:
        assert file in created_file_names


def test_smoke_elicit_run_tiny_gpt2_eigen(tmp_path: Path):
    """
    Currently this test fails with
    u -= torch.einsum("...ij,...i->...j", V[..., :k, :], proj)
    V[..., k, :] = F.normalize(u, dim=-1)
    ~~~~~~~~~ <--- HERE

    u[:] = torch.einsum("...ij,...j->...i", A, V[..., k, :])

    RuntimeError: select(): index 1 out of range for tensor of size [1, 2]
    at dimension 0
    """
    model_path = "sshleifer/tiny-gpt2"
    dataset_name = "imdb"
    config = RunConfig(
        data=ExtractionConfig(
            model=model_path,
            prompts=PromptConfig(dataset=dataset_name, max_examples=[10]),
            # run on all layers, tiny-gpt only has 2 layers
        ),
        net=EigenReporterConfig(),
    )
    train(config, tmp_path)
    # get the files in the tmp_path
    files: Path = list(tmp_path.iterdir())
    created_file_names = {file.name for file in files}
    expected_files = ["cfg.yaml", "metadata.yaml", "lr_models", "reporters", "eval.csv"]
    for file in expected_files:
        assert file in created_file_names
