import os
from dataclasses import dataclass, field
from pathlib import Path

from elk.plotting.visualize import visualize_sweep


@dataclass
class Plot:
    sweep: list[str] = field(default_factory=list)

    def execute(self):
        sweeps_path = Path.home() / "elk-reporters" / "sweeps"
        # in sweeps_path find the most recent sweep
        sweep = max(sweeps_path.iterdir(), key=os.path.getctime)
        if self.sweep:
            sweep = sweeps_path / self.sweep[0]
            if not sweep.exists():
                print(f"No sweep with name {self.sweep[0]} found in {sweeps_path}")
                return
        if len(self.sweep) > 1:
            print(
                f"""{len(self.sweep)} paths specified.
                Only one sweep is supported at this time."""
            )
        else:
            visualize_sweep(sweep)  # TODO support more than one sweep
