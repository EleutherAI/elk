from extraction.parser import get_args
from extraction.load_utils import (
    create_setname_to_promptframe,
    get_num_templates_per_dataset,
)
from extraction.extraction import extract_hiddens
from extraction.save_utils import (
    save_hidden_state_to_np_array,
    save_records_to_csv,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


if __name__ == "__main__":
    args = get_args()

    # AutoModel should do the right thing here in nearly all cases. We don't actually
    # care what head the model has, since we are just extracting hidden states.
    print(f"Loading model '{args.model}'...")
    model = AutoModel.from_pretrained(args.model, torch_dtype="auto").to(args.device)
    print(f"Done. Model class: '{model.__class__.__name__}'")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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

    for dataset_name, dataframe in tqdm(
        name_to_dataframe.items(),
        desc="Extracting",
    ):
        # TODO: Could use further cleanup
        hidden_state = extract_hiddens(args, model, tokenizer, dataframe)
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
