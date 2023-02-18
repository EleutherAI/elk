from argparse import ArgumentParser
from pathlib import Path
from ..files import elk_cache_dir


def get_evaluate_parser():
    parser = ArgumentParser(add_help=False)
    add_eval_args(parser)
    return parser


def add_eval_args(parser):
    parser.add_argument(
        "name",
        type=str,
        help="Name of the experiment containing" 
        "the reporters you want to evaluate.",
    )
    parser.add_argument(
        "reporter_name",
        type=str,
        help="Name of the reporter subfolder"
        "to save the trained reporters to.",
    )
    parser.add_argument(
        "--hidden-states",
        nargs="+",
        type=str,
        help="Name of the experiment containing the"
        "hidden states you want to evaluate the reporters on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="PyTorch device to use. Default is cuda:0 if available.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="meanonly",
        choices=("legacy", "elementwise", "meanonly"),
        help="Normalization method to use for CCS.",
    )
