from elk.training.parser import get_training_parser
from argparse import ArgumentParser


def get_rccs_training_parser() -> ArgumentParser:
    parser = get_training_parser()
    parser.add_argument(
        "--num-iterations",
        type=int,
        help="Number of recursive iterations in RCCS",
    )
    return parser
