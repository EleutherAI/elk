from argparse import ArgumentParser


def get_training_parser(name=True) -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    if name:
        parser.add_argument("name", type=str, help="Name of the experiment")
    add_train_args(parser)
    return parser


def add_train_args(parser: ArgumentParser):
    parser.add_argument(
        "--reporter-name",
        type=str,
        help="Name of the reporter subfolder" "to save the trained reporters to.",
        default=None,
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
