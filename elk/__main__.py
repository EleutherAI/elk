"""Main entry point for `elk`."""

from dataclasses import dataclass
from typing import Union
from simple_parsing import ArgumentParser
from elk.evaluation.evaluate import Eval
from elk.extraction.extraction import Extract

from elk.training.train import Elicit


@dataclass
class CommandRunner:
    """Some top-level command"""

    command: Union[Elicit, Eval, Extract]

    def execute(self):
        return self.command.execute()


def run():
    parser = ArgumentParser(add_help=False)
    parser.add_arguments(CommandRunner, dest="run")
    args = parser.parse_args()
    run: CommandRunner = args.run
    run.execute()


if __name__ == "__main__":
    run()
