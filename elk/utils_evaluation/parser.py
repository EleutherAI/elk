from pathlib import Path
import argparse
import json


def get_args(default_config_path=Path(__file__).parent / "default_config.json"):
    with open(default_config_path, "r") as f:
        default_config = json.load(f)

    datasets = default_config["datasets"]
    model_shortcuts = default_config["model_shortcuts"]
    prefix = default_config["prefix"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--prefix", default="normal", choices=prefix)
    parser.add_argument("--dataset", default=datasets[0])
    parser.add_argument("--num_data", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="concat",
        choices=["minus", "concat"],
        help="How you combine h^+ and h^-.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default="evaluation_results",
        help="Where the CSV and params are saved",
    )
    parser.add_argument(
        "--trained_models_path",
        type=Path,
        default="trained",
        help="Where to save the CCS and logistic regression models",
    )
    parser.add_argument(
        "--hidden_states_directory",
        type=Path,
        default="generation_results",
        help="Where the hidden states and zero-shot accuracy are loaded.",
    )
    parser.add_argument("--language_model_type", type=str, default="encoder")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        help="PyTorch device to use. Default is cuda:0 if available.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=("adam", "lbfgs"),
        help="Optimizer for CCS. Should be adam or lbfgs.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help=(
            "Weight decay for CCS when using Adam. Used as L2 regularization in LBFGS."
        ),
    )
    args = parser.parse_args()

    # Dereference shortcut
    args.model = model_shortcuts.get(args.model, args.model)

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.language_model_type == "decoder" and args.layer < 0:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model)
        args.layer += getattr(config, "num_layers", config.num_hidden_layers)

    return args
