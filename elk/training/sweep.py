from dataclasses import dataclass

from ..extraction import Extract, PromptConfig
from ..files import elk_reporter_dir, memorably_named_dir
from .train import Elicit


@dataclass
class Sweep:
    models: list[str]
    datasets: list[str]

    def __post_init__(self):
        if not self.models:
            raise ValueError("No models specified")
        if not self.datasets:
            raise ValueError("No datasets specified")

    def execute(self):
        M, D = len(self.models), len(self.datasets)
        print(f"Starting sweep over {M} models and {D} datasets ({M * D} runs))")

        root_dir = elk_reporter_dir() / "sweeps"
        sweep_dir = memorably_named_dir(root_dir)
        print(f"Saving sweep results to \033[1m{sweep_dir}\033[0m")  # bold

        for i, model_str in enumerate(self.models):
            # Magenta color for the model name
            print(f"\n\033[35m===== {model_str} ({i + 1} of {M}) =====\033[0m")

            for dataset_str in self.datasets:
                out_dir = sweep_dir / model_str / dataset_str

                Elicit(
                    data=Extract(
                        model=model_str,
                        prompts=PromptConfig(
                            datasets=[dataset_str],
                        ),
                    ),
                    out_dir=out_dir,
                ).execute()
