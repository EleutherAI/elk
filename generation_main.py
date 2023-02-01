import time
from utils_generation.parser import get_args
from utils_generation.load_utils import load_model, put_model_on_device, load_tokenizer, create_name_to_dataframe, get_num_templates_per_dataset
from utils_generation.generation import calculate_hidden_state
from utils_generation.save_utils import save_hidden_state_to_np_array, save_records_to_csv, print_elapsed_time
from tqdm import tqdm 
import torch


if __name__ == "__main__":
    print("\n\n-------------------------------- Starting Program --------------------------------\n\n")
    start = time.time()
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    print("\n\n-------------------------------- Args --------------------------------\n\n")
    args = get_args()
    for key in list(vars(args).keys()):
        print(f"{key}: {vars(args)[key]}")
    

    print("\n\n--------------------------------  Setting up model and tokenizer --------------------------------\n\n")
    print(f"Loading model: model name = {args.model} at cache_dir = {args.cache_dir}")
    model = load_model(mdl_name=args.model, cache_dir=args.cache_dir)
    
    print(f"Linish loading model to memory. Now start loading to accelerator (gpu or mps). parallelize = {args.parallelize is True}")
    model = put_model_on_device(model, parallelize=args.parallelize, device=args.model_device)
    
    print(f"Loading tokenizer for: model name = {args.model} at cache_dir = {args.cache_dir}")
    tokenizer = load_tokenizer(mdl_name=args.model, cache_dir=args.cache_dir)

    print("\n\n-------------------------------- Loading datasets and calculating hidden states --------------------------------\n\n")
    num_templates_per_dataset = get_num_templates_per_dataset(args.datasets)
    name_to_dataframe = create_name_to_dataframe(args.data_base_dir, args.datasets, num_templates_per_dataset, args.num_data, 
                                              tokenizer, args.save_base_dir, args.model, args.prefix, args.token_place, args.reload_data)
    
    print("\n\n-------------------------------- Generating hidden states --------------------------------\n\n")
    with torch.no_grad():
        for dataset_name, dataframe in tqdm(name_to_dataframe.items(), desc='Iterating over dataset-prompt combinations:'):
            hidden_state = calculate_hidden_state(args, model, tokenizer, dataframe, args.model)
            #TODO: clean up this ['0','1'] thing
            save_hidden_state_to_np_array(hidden_state, dataset_name, ['0','1'], args)
        
        records = []
        for dataset_name, dataframe in name_to_dataframe.items():
            records.append({
                "model": args.model,
                "dataset": dataset_name,
                "prefix": args.prefix,
                "tag": args.tag,
                "cal_hiddenstates": bool(args.cal_hiddenstates),
                "population": len(dataframe)
                })
        save_records_to_csv(records, args)
                    
    print_elapsed_time(start, args.prefix, name_to_dataframe)
    
    print("-------------------------------- Finishing Program --------------------------------")
