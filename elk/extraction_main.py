from elk.extraction.parser import get_args
from elk.extraction.load_utils import (
    load_model,
    put_model_on_device,
    load_datasets,
)
from elk.extraction.extraction import create_records, create_hiddenstates
from transformers import AutoTokenizer
from tqdm import tqdm

if __name__ == "__main__":
    # get args
    args = get_args()
    print(f"loading model: model name = {args.model}")
    model = load_model(mdl_name=args.model)

    print(
        "finish loading model to memory. Now start loading to accelerator (gpu or"
        f" mps). parallelize = {args.parallelize is True}"
    )
    model = put_model_on_device(model, parallelize=args.parallelize, device=args.device)

    print(f"loading tokenizer for: model name = {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(
        "\n\n-------------------------------- Loading datasets and calculating hidden"
        " states --------------------------------\n\n"
    )
    all_prefixes = args.prefix
    for prefix in tqdm(all_prefixes, desc="Iterating over prefixes:", position=0):
        args.prefix = prefix
        # load datasets and save if possible
        name_to_dataframe = load_datasets(args, tokenizer)

        # For each frame, generate the hidden states and save to directories
        print(
            "\n\n-------------------------------- Generating hidden states"
            " --------------------------------\n\n"
        )
        create_hiddenstates(model, tokenizer, name_to_dataframe, args)
        create_records(model, tokenizer, name_to_dataframe, args)
