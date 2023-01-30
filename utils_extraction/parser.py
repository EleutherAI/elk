import argparse
import json

def get_args(default_config_path = "default_config.json"):

    with open(default_config_path, "r") as f:
        default_config = json.load(f)
    datasets = default_config["datasets"]
    models = default_config["models"]
    prefix = default_config["prefix"]
    models_layer_num = default_config["models_layer_num"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, choices=models)
    parser.add_argument("--prefix", nargs="+", default = ["normal"], choices = prefix)
    parser.add_argument("--dataset", default = datasets[0])
    parser.add_argument("--test", type = str, default = "testall", choices = ["testone", "testall"])
    parser.add_argument("--num_data", type = int, default = 10) # default = 1000
    parser.add_argument("--methods", nargs="+", default = ["LR", "Prob"])
    parser.add_argument("--mode", type = str, default = "auto", choices = ["auto", "minus", "concat"], help = "How you combine h^+ and h^-.")
    parser.add_argument("--save_dir", type = str, default = "extraction_results", help = "where the csv and params are saved")
    parser.add_argument("--hidden_states_directory", type = str, default = "./generation_results", help = "Where the hidden states and zero-shot accuracy are loaded.")
    parser.add_argument("--language_model_type", type = str, default = "encoder")
    parser.add_argument("--layer", type = int, default = -1)
    parser.add_argument("--zero", type = str, default = "results")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--prompt_save_level", default = "all", choices = ["single", "all"])
    parser.add_argument("--model_device", type=str, default="cuda", help="What device to load the model onto: CPU or GPU or MPS.")
    args = parser.parse_args()

    assert args.test != "testone", NotImplementedError("Current extraction program does not support applying method on prompt-specific level.")

    if args.language_model_type == "decoder" and args.layer < 0:
        args.language_model_type += models_layer_num[args.model]

    return args