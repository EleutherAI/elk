import time
from elk.extraction.parser import get_args
from elk.extraction.load_utils import (
    load_model,
    put_model_on_device,
    load_tokenizer,
    load_datasets,
)
from elk.extraction.extraction import create_records, create_hiddenstates
from tqdm import tqdm

if __name__ == "__main__":
    print(
        "\n\n-------------------------------- Starting Program"
        " --------------------------------\n\n"
    )
    start = time.time()
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # get args
    args = get_args()

    # load model and tokenizer (put model on hardware accelearator if possible)
    print(
        "\n\n--------------------------------  Setting up model and tokenizer"
        " --------------------------------\n\n"
    )
    print(f"loading model: model name = {args.model}")
    model = load_model(mdl_name=args.model)

    print(
        "finish loading model to memory. Now start loading to accelerator (gpu or"
        f" mps). parallelize = {args.parallelize is True}"
    )
    model = put_model_on_device(model, parallelize=args.parallelize, device=args.device)

    print(f"loading tokenizer for: model name = {args.model}")
    tokenizer = load_tokenizer(mdl_name=args.model)

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

        total_samples = sum(
            [len(dataframe) for dataframe in name_to_dataframe.values()]
        )
        end = time.time()
        elapsed_minutes = round((end - start) / 60, 1)
        print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        print(
            f"Prefix used: {prefix}, applied to {len(name_to_dataframe)} datasets,"
            f" {total_samples} samples in total, and took {elapsed_minutes} minutes."
        )
        print("\n\n---------------------------------------\n\n")

    print(
        "-------------------------------- Finishing Program"
        " --------------------------------"
    )
