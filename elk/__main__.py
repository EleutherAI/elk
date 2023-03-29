"""Main entry point for `elk`."""

from dataclasses import dataclass
from typing import Union
import torch.multiprocessing as mp
mp.set_start_method("spawn")

from simple_parsing import ArgumentParser

from elk.evaluation.evaluate import Eval
from elk.extraction.extraction import Extract
from elk.training.train import Elicit


@dataclass
class Command:
    """Some top-level command"""

    command: Union[Elicit, Eval, Extract]

    def execute(self):
        return self.command.execute()


def run():
    """
    Sets the spawn start method for multiprocessing
    This is so that we can use CUDA properly in the child processes
    """
    print(f"Set start method to {mp.get_start_method()}")
    parser = ArgumentParser(add_help=False)
    parser.add_arguments(Command, dest="run")
    args = parser.parse_args()
    run: Command = args.run
    run.execute()


if __name__ == "__main__":
    run()
