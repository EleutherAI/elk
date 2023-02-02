import argparse
import json
from pathlib import Path

def get_args(default_config_path= Path(__file__).parent / "default_config.json"):

    with open(default_config_path, "r") as f:
        default_config = json.load(f)
    datasets = default_config["datasets"]
    models = default_config["models"]
    prefix = default_config["prefix"]
    models_layer_num = default_config["models_layer_num"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=models)
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
        help="where the csv and params are saved",
    )
    parser.add_argument(
        "--trained_models_path",
        type=Path,
        default="trained",
        help="where to save the models trained via ccs and logistic regression",
    )
    parser.add_argument(
        "--hidden_states_directory",
        type=Path,
        default="generation_results",
        help="Where the hidden states and zero-shot accuracy are loaded.",
    )
    parser.add_argument("--language_model_type", type=str, default="encoder")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--zero", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda",
        help="What device to load the model onto: CPU or GPU or MPS.",
    )
    args = parser.parse_args()

    if args.language_model_type == "decoder" and args.layer < 0:
        args.language_model_type += models_layer_num[args.model]

    return args
