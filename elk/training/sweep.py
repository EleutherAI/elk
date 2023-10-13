from dataclasses import InitVar, dataclass, field, replace

import numpy as np
import torch
from datasets import get_dataset_config_info
from transformers import AutoConfig

from ..evaluation import Eval
from ..files import memorably_named_dir, sweeps_dir
from ..plotting.visualize import visualize_sweep
from ..training.eigen_reporter import EigenFitterConfig
from ..utils import colorize
from ..utils.constants import BURNS_DATASETS
from .train import Elicit


def assert_models_exist(model_names):
    for model_name in model_names:
        AutoConfig.from_pretrained(model_name)


def assert_datasets_exist(dataset_names):
    for dataset_name in dataset_names:
        ds_name, _, config_name = dataset_name.partition(":")
        get_dataset_config_info(ds_name, config_name=config_name)


@dataclass
class Sweep:
    models: list[str]
    """List of Huggingface model strings to sweep over."""

    datasets: list[str]
    """List of dataset strings to sweep over. Each dataset string can contain
    multiple datasets, separated by plus signs. For example, "sst2+imdb" will
    pool SST-2 and IMDB together."""

    add_pooled: InitVar[bool] = False
    """Whether to add a dataset that pools all of the other datasets together."""

    hparam_step: float = -1.0
    """The step size for hyperparameter sweeps. Performs a 2D
    sweep over a and b in (var_weight, inv_weight, neg_cov_weight) = (a, 1 - b, b)
    If negative, no hyperparameter sweeps will be performed. Only valid for Eigen."""

    skip_transfer_eval: bool = False
    """Whether to perform transfer eval on every pair of datasets."""

    visualize: bool = False
    """Whether to generate visualizations of the results of the sweep."""

    name: str | None = None

    # A bit of a hack to add all the command line arguments from Elicit
    run_template: Elicit = field(default_factory=Elicit.default)

    def __post_init__(self, add_pooled: bool):
        if not self.datasets:
            raise ValueError("No datasets specified")
        if not self.models:
            raise ValueError("No models specified")
        # can only use hparam_step if we're using an eigen net
        if self.hparam_step > 0 and not isinstance(
            self.run_template.net, EigenFitterConfig
        ):
            raise ValueError("Can only use hparam_step with EigenFitterConfig")
        elif self.hparam_step > 1:
            raise ValueError("hparam_step must be in [0, 1]")

        # Check for the magic dataset "burns" which is a shortcut for all of the
        # datasets used in Burns et al., except Story Cloze, which is not available
        # on the Huggingface Hub.
        if "burns" in self.datasets:
            self.datasets.remove("burns")
            self.datasets.extend(BURNS_DATASETS)
            print(
                "Interpreting `burns` as all datasets used in Burns et al. (2022) "
                "available on the HuggingFace Hub"
            )

        # Remove duplicates just in case
        self.datasets = sorted(set(self.datasets))

        # Add an additional dataset that pools all of the datasets together.
        if add_pooled:
            self.datasets.append("+".join(self.datasets))

    def execute(self):
        M, D = len(self.models), len(self.datasets)
        print(f"Starting sweep over {M} models and {D} datasets ({M * D} runs)")
        assert_models_exist(self.models)
        print(f"Models: {self.models}")
        assert_datasets_exist(self.datasets)
        print(f"Datasets: {self.datasets}")

        root_dir = sweeps_dir()
        sweep_dir = root_dir / self.name if self.name else memorably_named_dir(root_dir)
        print(f"Saving sweep results to \033[1m{sweep_dir}\033[0m")  # bold

        # Each dataset string can contain multiple datasets, delimited by plus; this
        # indicates that the component datasets will be pooled together for training.
        # For example, we might be sweeping over ["amazon_polarity", "imdb+sst2"]. For
        # transfer eval, we want to split "imdb+sst2" into ["imdb", "sst2"] and then
        # flatten the list, yielding ["amazon_polarity", "imdb", "sst2"].
        eval_datasets = sorted(
            {
                ds.strip()
                for dataset_str in self.datasets
                for ds in dataset_str.split("+")
            }
        )

        step = self.hparam_step
        weights = np.arange(0.0, 1.0 + step, step) if step > 0 else [None]

        for i, model in enumerate(self.models):
            print(colorize(f"===== {model} ({i + 1} of {M}) =====", "magenta"))

            for dataset_str in self.datasets:
                # Allow for multiple datasets to be specified in a single string with
                # plus signs. This means we can pool datasets together inside of a
                # single sweep.
                train_datasets = tuple(ds.strip() for ds in dataset_str.split("+"))

                for var_weight in weights:
                    for neg_cov_weight in weights:
                        out_dir = sweep_dir / model / dataset_str

                        data = replace(
                            self.run_template.data, model=model, datasets=train_datasets
                        )
                        run = replace(self.run_template, data=data, out_dir=out_dir)
                        if var_weight is not None and neg_cov_weight is not None:
                            assert isinstance(run.net, EigenFitterConfig)
                            run.net.var_weight = var_weight
                            run.net.neg_cov_weight = neg_cov_weight

                            # Add hyperparameter values to output directory if needed
                            assert run.out_dir is not None
                            run.out_dir /= f"var_weight={var_weight:.2f}"
                            run.out_dir /= f"neg_cov_weight={neg_cov_weight:.2f}"

                        try:
                            run.execute()
                        except torch.linalg.LinAlgError as e:
                            print(colorize(f"LinAlgError: {e}", "red"))
                            continue

                        if not self.skip_transfer_eval:
                            if len(eval_datasets) > 1:
                                print(colorize("== Transfer eval ==", "green"))

                            # Now evaluate the reporter on the other datasets
                            for eval_dataset in eval_datasets:
                                # We already evaluated on this one during training
                                if eval_dataset in train_datasets:
                                    continue

                                assert run.out_dir is not None
                                eval = Eval(
                                    data=replace(
                                        run.data, model=model, datasets=(eval_dataset,)
                                    ),
                                    source=run.out_dir,
                                    out_dir=run.out_dir / "transfer" / eval_dataset,
                                    num_gpus=run.num_gpus,
                                    min_gpu_mem=run.min_gpu_mem,
                                    skip_supervised=run.supervised == "none",
                                )
                                eval.execute(highlight_color="green")

        if self.visualize:
            visualize_sweep(sweep_dir)
