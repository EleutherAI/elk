from argparse import ArgumentParser
from pathlib import Path
from ..files import elk_cache_dir


def get_evaluate_parser():
    parser = ArgumentParser(add_help=False)
    add_eval_args(parser)
    return parser


def add_eval_args(parser):
    parser.add_argument(
        "ccs_models_path",
        type=Path,
        help="Path for hidden states you want to evaluate the model on.",
    )
    parser.add_argument(
        "--hidden-states-path",
        nargs="+",
        type=Path,
        help="Path for hidden states you want to evaluate the model on.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=elk_cache_dir() / "eval_results",
        help="Output path for the evaluation results.",
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
