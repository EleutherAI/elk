from dataclasses import InitVar, dataclass

from ..extraction import Extract, PromptConfig
from ..files import elk_reporter_dir, memorably_named_dir
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

        for i, model_str in enumerate(self.models):
            # Magenta color for the model name
            print(f"\n\033[35m===== {model_str} ({i + 1} of {M}) =====\033[0m")

            for dataset_str in self.datasets:
                out_dir = sweep_dir / model_str / dataset_str

                # Allow for multiple datasets to be specified in a single string with
                # plus signs. This means we can pool datasets together inside of a
                # single sweep.
                datasets = [ds.strip() for ds in dataset_str.split("+")]
                Elicit(
                    data=Extract(
                        model=model_str,
                        prompts=PromptConfig(
                            datasets=datasets,
                        ),
                    ),
                    out_dir=out_dir,
                ).execute()
