from copy import deepcopy
from dataclasses import InitVar, dataclass

from ..evaluation.evaluate import Eval
from ..extraction import Extract, PromptConfig
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
            prompts=PromptConfig(datasets=["<placeholder>"]),
        )
    )

    def __post_init__(self, add_pooled: bool):
        if not self.datasets:
            raise ValueError("No datasets specified")
        if not self.models:
            raise ValueError("No models specified")

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

        for i, model_str in enumerate(self.models):
            # Magenta color for the model name
            print(f"\n\033[35m===== {model_str} ({i + 1} of {M}) =====\033[0m")

            for dataset_str in self.datasets:
                out_dir = sweep_dir / model_str / dataset_str

                # Allow for multiple datasets to be specified in a single string with
                # plus signs. This means we can pool datasets together inside of a
                # single sweep.
                train_datasets = [ds.strip() for ds in dataset_str.split("+")]

                run = deepcopy(self.run_template)
                run.data.model = model_str
                run.data.prompts.datasets = train_datasets
                run.out_dir = out_dir
                run.execute()

                if len(eval_datasets) > 1:
                    print(colorize("== Transfer eval ==", "green"))

                # Now evaluate the reporter on the other datasets
                for eval_dataset in eval_datasets:
                    # We already evaluated on this one during training
                    if eval_dataset in train_datasets:
                        continue

                    data = deepcopy(run.data)
                    data.model = model_str
                    data.prompts.datasets = [eval_dataset]

                    eval = Eval(
                        data=data,
                        source=str(run.out_dir),
                        out_dir=out_dir,
                    )
                    eval.execute(highlight_color="green")
