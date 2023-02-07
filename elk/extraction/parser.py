import json
from argparse import ArgumentParser
from pathlib import Path


def get_args():
    default_config_path = Path(__file__).parent.parent / "default_config.json"
    with open(default_config_path, "r") as f:
        default_config = json.load(f)
        model_shortcuts = default_config["model_shortcuts"]

    parser = get_parser()
    args = parser.parse_args()

    # Dereference shortcut
    args.model = model_shortcuts.get(args.model, args.model)

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    for key in list(vars(args).keys()):
        print("{}: {}".format(key, vars(args)[key]))

    return args


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace name of model from which to extract hidden states.",
    )
    parser.add_argument(
        "dataset",
        nargs="+",
        help="HuggingFace dataset you want to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="PyTorch device to use. Default is cuda:0 if available.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        type=str,
        help="Column of the dataset to use as the label. Default is 'label'.",
    )
    parser.add_argument(
        "--max-examples",
        default=1000,
        type=int,
        help="Maximum number of examples to use from each dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment. If not provided, a memorable name of the form "
        "`objective-ramanujan` will be generated.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="",
        help=(
            "Suffix to append to the prompt after the answer. This sometimes improves"
            " performance for autoregressive models."
        ),
    )
    parser.add_argument(
        "--token-loc",
        type=str,
        default="last",
        help=(
            "Determine which token's hidden states will be extractd. Can be `first` or"
            " `last` or `average`."
        ),
    )
    parser.add_argument(
        "--use-encoder-states",
        action="store_true",
        help=(
            "Whether to extract encoder hidden states in encoder-decoder models, by"
            " including the answer in the input to the encoder. By default we pass the"
            " question to the encoder and the answer to the decoder, extracting the"
            " decoder hidden state. This is closer to the pretraining setting for most"
            " encoder-decoder models, and it allows for reusing the encoder hidden"
            " states across different answers to the same question."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layers to extract hiddens from. If None, extract from all layers.",
    )

    return parser
