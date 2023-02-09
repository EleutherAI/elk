import argparse
import json
from transformers import AutoConfig, PretrainedConfig
from pathlib import Path


def get_args():
    default_config_path = Path(__file__).parent.parent / "default_config.json"
    with open(default_config_path, "r") as f:
        default_config = json.load(f)
        datasets = default_config["datasets"]
        prefix = default_config["prefix"]
        model_shortcuts = default_config["model_shortcuts"]

    parser = get_extraction_parser()
    args = parser.parse_args()

    # Default to CUDA iff available
    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.datasets == ["all"]:
        args.datasets = datasets
    else:
        for w in args.datasets:
            assert w in datasets, NotImplementedError(
                "Dataset {} not  in {}. Please check the name of the dataset!".format(
                    w, default_config_path
                )
            )

    for prefix in args.prefix:
        assert prefix in prefix, NotImplementedError(
            "Invalid prefix name {}. Please check your prefix name. To add new prefix,"
            " please mofidy `extraction/prompts.json` \
                and new prefix in {}.json.".format(
                prefix, default_config_path
            )
        )

    args.model = model_shortcuts.get(args.model, args.model)
    config = AutoConfig.from_pretrained(args.model)
    assert isinstance(config, PretrainedConfig)

    num_layers = getattr(config, "num_layers", config.num_hidden_layers)
    assert isinstance(num_layers, int)

    if args.use_encoder_states and not config.is_encoder_decoder:
        raise ValueError(
            "--use_encoder_states is only compatible with encoder-decoder models."
        )

    for key in list(vars(args).keys()):
        print("{}: {}".format(key, vars(args)[key]))

    return args


def get_extraction_parser():
    parser = argparse.ArgumentParser()

    # datasets loading
    parser.add_argument(
        "--data-base-dir",
        type=Path,
        default="datasets/complete_ten",
        help=(
            "The base dir of all datasets (csv files) you want to generate hidden"
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
        "--device",
        type=str,
        help="PyTorch device to use. Default is cuda:0 if available.",
    )
    parser.add_argument(
        "--trained-models-path",
        type=Path,
        default="trained",
        help="Where to save the ccs model and logisitc regression model.",
    )

    # datasets processing
    parser.add_argument(
        "--prefix",
        type=str,
        nargs="+",
        default=["normal"],
        help=(
            "The name of prefix added before the question. normal means no index. You"
            " can go to `extraction/prompts.json` to add new prompt."
        ),
    )
    parser.add_argument(
        "--num-data",
        nargs="+",
        default=[1000],
        help=(
            "number of data points you want to use in each datasets. If one integer is"
            " provide, if will be extended to a list with the same length as"
            " `datasets`. If the size of datasets are no enough, will use all the data"
            " points."
        ),
    )
    parser.add_argument(
        "--reload-data",
        action="store_true",
        help=(
            "Whether to use the old version of datasets if there exists one. Using"
            " `reload_data` will let the program reselect data points from the"
            " datasets."
        ),
    )
    parser.add_argument(
        "--prompt-idx",
        nargs="+",
        default=[0],
        help="The indices of prompt you want to use.",
    )

    # extraction & zero-shot accuracy calculation
    parser.add_argument(
        "--cal-zeroshot",
        type=int,
        default=1,
        help="Whether to calculate the zero-shot accuracy.",
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
            "Determine which token's hidden states will be generated. Can be `first` or"
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
        "--tag",
        type=str,
        default="",
        help="Tag added as the suffix of the directory.",
    )
    parser.add_argument(
        "--save-base-dir",
        type=Path,
        default="extraction_results",
        help="The base dir where you want to save the directories of hidden states.",
    )
    parser.add_argument(
        "--save-csv-name",
        type=str,
        default="results",
        help="Name of csv that store all running records.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Which layers to extract hiddens from. If None, extract from all layers.",
    )
    parser.add_argument(
        "--print-more", action="store_true", help="Whether to print more."
    )

    return parser
