"""Main entry point for `elk`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from simple_parsing import ArgumentParser

from elk.evaluation.evaluate import Eval, evaluate_reporters

from .extraction import ExtractionConfig, extract
from .training.train import Elicit


@dataclass
class Run:
    """Some top-level command"""
    command: Union[Elicit, Eval] 

    def execute(self):
        return self.command.execute() # type: ignore

def run():
    parser = ArgumentParser(add_help=False)
    parser.add_arguments(Run, dest="run")
    args = parser.parse_args()
    run: Run = args.run
    run.execute()


if __name__ == "__main__":
    run()
