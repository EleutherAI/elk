import json
from argparse import ArgumentParser
from pathlib import Path


def get_args():
    default_config_path = Path(__file__).parent.parent / "default_config.json"
    with open(default_config_path, "r") as f:
        default_config = json.load(f)

    args = get_parser().parse_args()

    # Dereference shortcut
    args.model = default_config["model_shortcuts"].get(args.model, args.model)

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="Name of the experiment")
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace name of model from which to extract hidden states.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="HuggingFace dataset you want to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="concat",
        choices=["minus", "concat"],
        help="How you combine h^+ and h^-.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default="evaluation_results",
        help="Where the CSV and params are saved",
    )
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)

    return parser
