"""Main entry point for `elk`."""

from dataclasses import dataclass

from simple_parsing import ArgumentParser

from elk.evaluation.evaluate import Eval
from elk.extraction.extraction import Extract
from elk.training.train import Elicit


@dataclass
class Command:
    """Some top-level command"""

    command: Elicit | Eval | Extract

    def execute(self):
        return self.command.execute()


def run():
    parser = ArgumentParser(add_help=False)
    parser.add_arguments(Command, dest="run")
    args = parser.parse_args()
    run: Command = args.run
    run.execute()


if __name__ == "__main__":
    run()
