"""Main entry point for `elk`."""

from .extraction import ExtractionConfig
from .files import args_to_uuid
from .list import list_runs
from .training import ReporterConfig
from contextlib import nullcontext, redirect_stdout
from simple_parsing import ArgumentParser
import logging
import warnings


def run():
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "extract",
        help="Extract hidden states from a model.",
    ).add_arguments(ExtractionConfig, dest="extraction_config")

    elicit_parser = subparsers.add_parser(
        "elicit",
        help=(
            "Extract and train a set of ELK reporters "
            "on hidden states from `elk extract`. "
        ),
        conflict_handler="resolve",
    )
    elicit_parser.add_arguments(ExtractionConfig, dest="extraction_config")
    elicit_parser.add_arguments(ReporterConfig, dest="reporter_config")

    subparsers.add_parser(
        "eval", help="Evaluate a set of ELK reporters generated by `elk train`."
    )
    subparsers.add_parser("list", help="List all cached runs.")
    args = parser.parse_args()

    # `elk list` is a special case
    if args.command == "list":
        list_runs(args)
        return

    from transformers import AutoConfig, PretrainedConfig

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

    # TODO: Remove this once the extraction refactor is finished
    if args.layers and args.layers != list(range(num_layers)):
        warnings.warn(
            "Warning: hidden states are not labeled by layer index, and reporter "
            "checkpoints generated by `elk elicit` will be incorrectly named; "
            "e.g. `layer_1` instead of `layer_2` for the 3rd transformer layer "
            "when `--layer-stride` is 2. This will be fixed in a future release."
        )

    # Import here and not at the top to speed up `elk list`
    from .extraction.extraction_main import run as run_extraction
    from .training.train import train
    import os
    import torch.distributed as dist

    # Check if we were called with torchrun or not
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        dist.init_process_group("nccl")
        local_rank = int(local_rank)

    with redirect_stdout(None) if local_rank else nullcontext():
        # Print CLI arguments to stdout
        for key, value in vars(args).items():
            print(f"{key}: {value}")

        if local_rank:
            logging.getLogger("transformers").setLevel(logging.CRITICAL)

        if args.command == "extract":
            run_extraction(args)
        elif args.command == "elicit":
            # The user can specify a name for the run, but by default we use the
            # MD5 hash of the arguments to ensure the name is unique
            if not args.name:
                args.name = args_to_uuid(args)

            try:
                train(args)
            except (EOFError, FileNotFoundError):
                run_extraction(args)

                # Ensure the extraction is finished before starting training
                if dist.is_initialized():
                    dist.barrier()

                train(args)

        elif args.command == "eval":
            # TODO: Implement evaluation script
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    run()
