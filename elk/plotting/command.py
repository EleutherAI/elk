import os
from dataclasses import dataclass, field
from pathlib import Path

from elk.plotting.visualize import visualize_sweep


@dataclass
class Plot:
    sweeps: list[Path] = field(default_factory=list)

    def execute(self):
        sweeps_path = Path.home() / "elk-reporters" / "sweeps"
        # in sweeps_path find the most recent sweep
        sweep = max(sweeps_path.iterdir(), key=os.path.getctime)
        if self.sweeps:
            sweep = self.sweeps[0]
        if len(self.sweeps) > 1:
            print(
                f"""{len(self.sweeps)} paths specified.
                Only one sweep is supported at this time."""
            )
        else:
            visualize_sweep(sweep)  # TODO support more than one sweep
