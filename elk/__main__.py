from elk.files import args_to_uuid, elk_cache_dir
from .extraction.extraction_main import run as run_extraction
from .extraction.parser import (
    add_saveable_args,
    add_unsaveable_args,
    get_extraction_parser,
)
from .training.parser import add_train_args, get_training_parser
from .training.train import train
from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoConfig, PretrainedConfig
import json


def sweep(model, datasets):
    """
    Train and evaluate ccs and lr model on a set of datasets.
    """
    # just run a for loop over the datasets... 
    # TODO: extract, train, eval



def run():
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "extract",
        help="Extract hidden states from a model.",
        parents=[get_extraction_parser()],
    )
    subparsers.add_parser(
        "train",
        help=(
            "Train a set of ELK probes on hidden states from `elk extract`. "
            "The first argument has to be the name you gave to the extraction."
        ),
        parents=[get_training_parser()],
    )
    subparsers.add_parser(
        "elicit",
        help=(
            "Extract and train a set of ELK probes "
            "on hidden states from `elk extract`. "
        ),
        parents=[get_extraction_parser(), get_training_parser(name=False)],
        conflict_handler="resolve",
    )

    subparsers.add_parser(
        "eval", help="Evaluate a set of ELK probes generated by `elk train`."
    )
    args = parser.parse_args()

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if model := getattr(args, "model", None):
        config_path = Path(__file__).parent / "default_config.json"
        with open(config_path, "r") as f:
            default_config = json.load(f)
            model_shortcuts = default_config["model_shortcuts"]

        # Dereference shortcut
        args.model = model_shortcuts.get(model, model)
        config = AutoConfig.from_pretrained(args.model)
        assert isinstance(config, PretrainedConfig)

        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        assert isinstance(num_layers, int)

        if args.layers and args.layer_stride > 1:
            raise ValueError(
                "Cannot use both --layers and --layer-stride. Please use only one."
            )
        elif args.layer_stride > 1:
            args.layers = list(range(0, num_layers, args.layer_stride))

    for key in list(vars(args).keys()):
        print("{}: {}".format(key, vars(args)[key]))

    # TODO: Implement the rest of the CLI
    if args.command == "extract":
        run_extraction(args)
    elif args.command == "train":
        train(args)
    elif args.command == "elicit":
        args.name = args_to_uuid(args)
        cache_dir = elk_cache_dir() / args.name
        if not cache_dir.exists():
            run_extraction(args)
        else:
            print(
                f"Cache dir \033[1m{cache_dir}\033[0m exists, "
                "skip extraction of hidden states"
            )  # bold
        train(args)
    elif args.command == "eval":
        sweep(args.model, args.datasets)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    run()
