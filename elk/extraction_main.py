from extraction.parser import get_args
from extraction.load_utils import (
    load_model,
    put_model_on_device,
    load_tokenizer,
    create_setname_to_promptframe,
    get_num_templates_per_dataset,
)
from extraction.extraction import calculate_hidden_state
from extraction.save_utils import (
    save_hidden_state_to_np_array,
    save_records_to_csv,
)
from tqdm import tqdm
import torch


if __name__ == "__main__":
    args = get_args()

    print(f"Loading model: model name = {args.model} at cache_dir = {args.cache_dir}")
    model = load_model(mdl_name=args.model, cache_dir=args.cache_dir)

    print(
        "Finished loading model to memory. Now start loading to accelerator (gpu or"
        f" mps). parallelize = {args.parallelize is True}"
    )
    model = put_model_on_device(model, parallelize=args.parallelize, device=args.device)

    print(
        f"Loading tokenizer for: model name = {args.model} at cache_dir ="
        f" {args.cache_dir}"
    )
    tokenizer = load_tokenizer(mdl_name=args.model, cache_dir=args.cache_dir)

    print("Loading datasets")
    num_templates_per_dataset = get_num_templates_per_dataset(args.datasets)
    name_to_dataframe = create_setname_to_promptframe(
        args.data_base_dir,
        args.datasets,
        num_templates_per_dataset,
        args.num_data,
        tokenizer,
        args.save_base_dir,
        args.model,
        args.prefix,
        args.token_place,
    )

    print("Extracting hidden states")
    with torch.no_grad():
        for dataset_name, dataframe in tqdm(
            name_to_dataframe.items(),
            desc="Iterating over dataset-prompt combinations:",
        ):
            # TODO: Could use further cleanup
            hidden_state = calculate_hidden_state(
                args, model, tokenizer, dataframe, args.model
            )
            # TODO: Clean up the ['0','1'] mess
            save_hidden_state_to_np_array(hidden_state, dataset_name, ["0", "1"], args)

        records = []
        for dataset_name, dataframe in name_to_dataframe.items():
            records.append(
                {
                    "model": args.model,
                    "dataset": dataset_name,
                    "prefix": args.prefix,
                    "tag": args.tag,
                    "cal_hiddenstates": bool(args.cal_hiddenstates),
                    "population": len(dataframe),
                }
            )
        save_records_to_csv(records, args)
