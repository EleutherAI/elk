import argparse
import json
from pathlib import Path

default_config_path = Path(__file__).parent.parent / "default_config.json"
json_dir = "./default_config"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
dataset_list = global_dict["datasets"]
registered_models = global_dict["models"]
registered_prefix = global_dict["prefix"]
models_layer_num = global_dict["models_layer_num"]


def get_args():
    parser = argparse.ArgumentParser()

    # datasets loading
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="datasets/complete_ten",
        help=(
            "The base dir of all datasets (csv files) you want to extract hidden"
            " states."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "List of name of datasets you want to use. Please make sure that the path"
            ' of file is like `data_base_dir / (name + ".csv")` for all name'
            " in datasets"
        ),
    )

    # models loading
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The model you want to use. Please use the model in huggingface and only"
            " leave the final path, i.e. for `allenai/unifiedqa-t5-11b`, only input"
            " `unifiedqa-t5-11b`."
        ),
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "Whether to parallelize models in multiple gpus. Please notice that at"
            " least one gpu must be provided by `CUDA_VISIBLE_DEVICES` or `os.environ`."
            " Using this args will help split the model equally in all gpus you"
            " provide."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="models",
        help="The path to save and load pretrained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="What device to load the model onto: CPU or GPU or MPS.",
    )

    # datasets processing
    parser.add_argument(
        "--prefix",
        type=str,
        default="normal",
        help=(
            "The name of prefix added before the question. normal means no index. You"
            " can go to `extraction/prompts.json` to add new prompt."
        ),
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=1000,
        help=(
            "number of data points you want to use in each datasets. If one integer is"
            " provide, if will be extended to a list with the same length as"
            " `datasets`. If the size of datasets are no enough, will use all the data"
            " points."
        ),
    )
    parser.add_argument(
        "--reload_data",
        action="store_true",
        help=(
            "Whether to use the old version of datasets if there exists one. Using"
            " `reload_data` will let the program reselect data points from the"
            " datasets."
        ),
    )
    parser.add_argument(
        "--prompt_idx",
        nargs="+",
        default=[0],
        help="The indexes of prompt you want to use.",
    )

    # extraction & zero-shot accuracy calculation
    parser.add_argument(
        "--cal_zeroshot",
        type=int,
        default=1,
        help="Whether to calculate the zero-shot accuracy.",
    )
    parser.add_argument(
        "--cal_hiddenstates",
        type=int,
        default=1,
        help="Whether to extract the hidden states.",
    )
    parser.add_argument(
        "--cal_logits",
        type=int,
        default=0,
        help=(
            "Whether to extract the logits of the token in which the prediction firstly"
            " differs."
        ),
    )
    parser.add_argument(
        "--token_place",
        type=str,
        default="last",
        help=(
            "Determine which token's hidden states will be extractd. Can be `first` or"
            " `last` or `average`."
        ),
    )
    parser.add_argument(
        "--states_location",
        type=str,
        default="null",
        choices=["encoder", "decoder", "null"],
        help=(
            "Whether to extract encoder hidden states or decoder hidden states."
            " Default is null, which will be extended to decoder when the model is gpt"
            " or encoder otherwise."
        ),
    )
    parser.add_argument(
        "--states_index",
        nargs="+",
        default=[-1],
        help=(
            "List of layer hidden states index to extract. -1 means the last layer."
            " For encoder, we will transform positive index into negative. For example,"
            " T0pp has 25 layer, indexed by 0, ..., 24. Index 20 will be transformed"
            " into -5. For decoder, index will instead be transform into non-negative"
            " value. For example, the last decoder layer will be 24 (rather than -1)."
            " The choice between encoder and decoder is specified by `states_location`."
            " For decoder, answer will be padded into token rather than into the input."
        ),
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Tag added as the suffix of the directory."
    )
    parser.add_argument(
        "--save_base_dir",
        type=str,
        default="extraction_results",
        help="The base dir where you want to save the directories of hidden states.",
    )
    parser.add_argument(
        "--save_csv_name",
        type=str,
        default="results",
        help="Name of csv that store all running records.",
    )
    parser.add_argument(
        "--save_all_layers",
        action="store_true",
        help=(
            "Whether to save the hidden states of all layers. Notice that this will"
            " increase the disk load significantly."
        ),
    )
    parser.add_argument(
        "--print_more", action="store_true", help="Whether to print more."
    )

    args = parser.parse_args()

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.datasets == ["all"]:
        args.datasets = dataset_list
    else:
        for dataset_name in args.datasets:
            assert dataset_name in dataset_list, NotImplementedError(
                f"Dataset {dataset_name} not registered in {json_dir}.json. Please"
                " check the name of the dataset!"
            )

    # if (args.cal_zeroshot or args.cal_logits) and "bert" in args.model:
    # Add features. Only forbid cal_logits for bert type model now
    if args.cal_logits and "bert" in args.model:
        raise NotImplementedError(
            f"You use {args.model}, but bert type models do not have standard logits."
            " Please set cal_logits to 0."
        )

    assert args.model in registered_models, NotImplementedError(
        f"You use model {args.model}, but it's not registered. For any new model,"
        " please make sure you implement the code in `load_utils` and `extraction`,"
        " and then register it in `parser.py`"
    )
    assert args.prefix in registered_prefix, NotImplementedError(
        f"Invalid prefix name {args.prefix}. Please check your prefix name. To add new"
        " prefix, please mofidy `extraction/prompts.json` and register new"
        f" prefix in {json_dir}.json."
    )

    # Set default states_location according to model type
    if args.states_location == "null":
        args.states_location = "decoder" if "gpt" in args.model else "encoder"
    if args.states_location == "encoder" and args.cal_hiddenstates:
        assert "gpt" not in args.model, ValueError(
            "GPT type model does not have encoder. Please set `states_location` to"
            " `decoder`."
        )
    if args.states_location == "decoder" and args.cal_hiddenstates:
        assert "bert" not in args.model, ValueError(
            "BERT type model does not have decoder. Please set `states_location` to"
            " `encoder`."
        )

    # Set index into int.
    for i in range(len(args.states_index)):
        pos_index = int(args.states_index[i]) % models_layer_num[args.model]
        # For decoder, the index lies in [0,layer_num)
        # For encoder, the index lies in [-layer_num, -1]
        if args.states_location == "decoder":
            args.states_index[i] = pos_index
        else:
            args.states_index[i] = pos_index - models_layer_num[args.model]

    return args
