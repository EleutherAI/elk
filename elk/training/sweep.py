from dataclasses import InitVar, dataclass, replace

from ..evaluation import Eval
from ..extraction import Extract
from ..files import elk_reporter_dir, memorably_named_dir
from ..utils import colorize
from .train import Elicit


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

    name: str | None = None

    # A bit of a hack to add all the command line arguments from Elicit
    run_template: Elicit = Elicit(
        data=Extract(
            model="<placeholder>",
            datasets=("<placeholder>",),
        )
    )

    def __post_init__(self, add_pooled: bool):
        if not self.datasets:
            raise ValueError("No datasets specified")
        if not self.models:
            raise ValueError("No models specified")

        # Check for the magic dataset "burns" which is a shortcut for all of the
        # datasets used in Burns et al., except Story Cloze, which is not available
        # on the Huggingface Hub.
        if "burns" in self.datasets:
            self.datasets.remove("burns")
            self.datasets.extend(
                [
                    "ag_news",
                    "amazon_polarity",
                    "dbpedia_14",
                    "glue:qnli",
                    "imdb",
                    "piqa",
                    "super_glue:boolq",
                    "super_glue:copa",
                    "super_glue:rte",
                ]
            )
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
        print(f"Models: {self.models}")
        print(f"Datasets: {self.datasets}")

        root_dir = elk_reporter_dir() / "sweeps"
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

        for i, model in enumerate(self.models):
            print(colorize(f"===== {model} ({i + 1} of {M}) =====", "magenta"))

            for dataset_str in self.datasets:
                out_dir = sweep_dir / model / dataset_str

                # Allow for multiple datasets to be specified in a single string with
                # plus signs. This means we can pool datasets together inside of a
                # single sweep.
                train_datasets = tuple(ds.strip() for ds in dataset_str.split("+"))

                data = replace(
                    self.run_template.data, model=model, datasets=train_datasets
                )
                run = replace(self.run_template, data=data, out_dir=out_dir)
                run.execute()

                if len(eval_datasets) > 1:
                    print(colorize("== Transfer eval ==", "green"))

                # Now evaluate the reporter on the other datasets
                for eval_dataset in eval_datasets:
                    # We already evaluated on this one during training
                    if eval_dataset in train_datasets:
                        continue

                    assert run.out_dir is not None
                    eval = Eval(
                        data=replace(run.data, model=model, datasets=(eval_dataset,)),
                        source=run.out_dir,
                        out_dir=out_dir / "transfer" / eval_dataset,
                        num_gpus=run.num_gpus,
                        min_gpu_mem=run.min_gpu_mem,
                    )
                    eval.execute(highlight_color="green")
