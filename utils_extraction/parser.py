import argparse
import json

def get_extraction_args(json_dir = "./registration"):

    with open(f"{json_dir}.json", "r") as f:
        global_dict = json.load(f)
    registered_dataset_list = global_dict["dataset_list"]
    registered_models = global_dict["registered_models"]
    registered_prefix = global_dict["registered_prefix"]
    models_layer_num = global_dict["models_layer_num"]


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, choices=registered_models)
    parser.add_argument("--prefix", nargs="+", default = ["normal"], choices = registered_prefix)
    parser.add_argument("--datasets", nargs="+", default = registered_dataset_list)
    parser.add_argument("--test", type = str, default = "testall", choices = ["testone", "testall"])
    parser.add_argument("--num_data", type = int, default = 10) # default = 1000
    parser.add_argument("--method_list", nargs="+", default = ["LR", "Prob"])
    parser.add_argument("--mode", type = str, default = "auto", choices = ["auto", "minus", "concat"], help = "How you combine h^+ and h^-.")
    parser.add_argument("--save_dir", type = str, default = "extraction_results", help = "where the csv and params are saved")
    parser.add_argument("--append", action="store_true", help = "Whether to append content in frame rather than rewrite.")
    parser.add_argument("--load_dir", type = str, default = "./generation_results", help = "Where the hidden states and zero-shot accuracy are loaded.")
    parser.add_argument("--location", type = str, default = "auto")
    parser.add_argument("--layer", type = int, default = -1)
    parser.add_argument("--zero", type = str, default = "results")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--prompt_save_level", default = "all", choices = ["single", "all"])
    parser.add_argument("--model_device", type=str, default="cuda", help="What device to load the model onto: CPU or GPU or MPS.")
    args = parser.parse_args()

    assert args.test != "testone", NotImplementedError("Current extraction program does not support applying method on prompt-specific level.")

    if args.location == "auto":
        args.location = "decoder" if "gpt" in args.model else "encoder"
    if args.location == "decoder" and args.layer < 0:
        args.layer += models_layer_num[args.model]

    return args