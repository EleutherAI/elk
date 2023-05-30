import shutil
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import field

from ..files import sweeps_dir
from ..utils import colorize
from .visualize import visualize_sweep


def pretty_error(msg):
    """Prints a pretty error message."""
    print(colorize("Error", "red") + f": {msg}")


@dataclass
class Plot:
    sweeps: list[Path] = field(positional=True, default_factory=list)
    """Names of the sweeps to plot. If empty, the most recent sweep is used."""

    overwrite: bool = False
    """Whether to overwrite existing plots."""

    def execute(self):
        root_dir = sweeps_dir()

        # If sweep is nonempty, get the paths to the specified sweeps.
        # If no sweep is specified, use the most recent one.
        if not self.sweeps:
            sweep_paths = [max(root_dir.iterdir(), key=lambda f: f.stat().st_ctime)]
            print(
                f"Reading most recent sweep from \033[1m{sweep_paths[0]}\033[0m"
            )  # bold
        else:
            sweep_paths = [root_dir / sweep for sweep in self.sweeps]

        for sweep_path in sweep_paths:
            if not sweep_path.exists():
                pretty_error(f"No sweep with name {{{sweep_path}}} found in {root_dir}")
            elif (sweep_path / "viz").exists() and not self.overwrite:
                pretty_error(
                    f"[blue]{sweep_path / 'viz'}[/blue] already exists. "
                    f"Use --overwrite to overwrite."
                )
            else:
                if self.overwrite:
                    shutil.rmtree(sweep_path / "viz")

                visualize_sweep(sweep_path)
