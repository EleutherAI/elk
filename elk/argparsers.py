"""Functions for constructing CLI argument parsers."""

from argparse import ArgumentParser


def get_extraction_parser():
    """Get parser for `elk extract`."""
    parser = ArgumentParser(add_help=False)
    add_saveable_args(parser)
    add_unsaveable_args(parser)
    return parser


def add_saveable_args(parser):
    """Add arguments that should be hashed into the UUID.

    These arguments are used to generate a unique ID for the extraction
    through the args_to_uuid function in elk/files.py.
    """
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
        "--label-column",
        default="label",
        type=str,
        help="Column of the dataset to use as the label. Default is 'label'.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum number of examples to use from each dataset.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="randomize",
        choices=("all", "randomize"),
        help=(
            "'all' means to use all prompts for every example, while 'randomize' means "
            "to assign a single random prompt to each data point."
        ),
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
            "Determine which token's hidden states will be extracted. Can be `first` or"
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
    parser.add_argument(
        "--layer-stride",
        default=1,
        type=int,
        help="Stride between layers to extract. Default is 1.",
    )


def add_unsaveable_args(parser):
    """Add arguments that should not be hashed into the UUID."""

    parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment. If not provided, a name as a md5 hash "
        "of the form c7f9cac6827745ec4d3ca2fcdbfde451 will be generated.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help=(
            "Fraction of `--max-examples` to use for testing. Ignored when "
            "`--max-examples` is None; in that case the whole test set is used."
        ),
    )
    parser.add_argument(
        "--balance",
        type=bool,
        default=False,
        help="Whether to balance the training set by class.",
    )
    return parser


def get_saveable_args(args):
    """Get all arguments that should be hashed into the UUID."""
    only_saveable_parser = ArgumentParser(add_help=False)
    add_saveable_args(only_saveable_parser)

    def get_names_from_action_group(group):
        return [action.dest for action in group._group_actions]

    only_saveable_args = sum(
        (
            get_names_from_action_group(group)
            for group in only_saveable_parser._action_groups
        ),
        [],
    )
    return {k: v for k, v in vars(args).items() if k in only_saveable_args}


def add_train_args(parser: ArgumentParser):
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=-1,
        help="Maximum number of GPUs to use. Default is -1, or no limit.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="meanonly",
        choices=("legacy", "elementwise", "meanonly"),
        help="Normalization method to use for CCS.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="default",
        choices=("default", "spherical", "zero"),
        help="Initialization for reporter.",
    )
    parser.add_argument(
        "--label-frac",
        type=float,
        default=0.0,
        help="Fraction of labeled data to use for training.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="squared",
        choices=("js", "squared"),
        help="Loss function used for reporter.",
    )
    parser.add_argument(
        "--num-tries",
        type=int,
        default=10,
        help="Number of random initializations to try.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="lbfgs",
        choices=("adam", "lbfgs"),
        help="Optimizer for reporter. Should be adam or lbfgs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip training the logistic regression baseline.",
    )
    parser.add_argument(
        "--supervised-weight",
        type=float,
        default=0.0,
        help="Weight of the supervised loss in the reporter objective.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help=(
            "Weight decay for reporter when using Adam. Used as L2 penalty in LBFGS."
        ),
    )
    return parser
