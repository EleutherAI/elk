import time
from utils_generation.parser import get_args
from utils_generation.load_utils import load_model, put_model_on_device, load_tokenizer, load_datasets
from utils_generation.generation import calculate_hidden_state
from utils_generation.save_utils import save_hidden_state_to_np_array, save_records_to_csv, print_elapsed_time
from tqdm import tqdm 
import torch


if __name__ == "__main__":
    print("\n\n-------------------------------- Starting Program --------------------------------\n\n")
    start = time.time()
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # get args
    args = get_args()

    # load model and tokenizer (put model on hardware accelearator if possible)
    print("\n\n--------------------------------  Setting up model and tokenizer --------------------------------\n\n")
    print(f"Loading model: model name = {args.model} at cache_dir = {args.cache_dir}")
    model = load_model(mdl_name=args.model, cache_dir=args.cache_dir)
    
    print(f"Linish loading model to memory. Now start loading to accelerator (gpu or mps). parallelize = {args.parallelize is True}")
    model = put_model_on_device(model, parallelize=args.parallelize, device = args.model_device)
    
    print(f"Loading tokenizer for: model name = {args.model} at cache_dir = {args.cache_dir}")
    tokenizer = load_tokenizer(mdl_name=args.model, cache_dir=args.cache_dir)

    print("\n\n-------------------------------- Loading datasets and calculating hidden states --------------------------------\n\n")
    all_prefixes = args.prefix

    for prefix in tqdm(all_prefixes, desc='Iterating over prefixes:', position=0):
        args.prefix = prefix
        # load datasets and save if possible
        # TODO: CLEAN THIS UP?
        name_to_dataframe = load_datasets(args, tokenizer)

        # For each frame, generate the hidden states and save to directories
        print("\n\n-------------------------------- Generating hidden states --------------------------------\n\n")
        with torch.no_grad():
            for dataset_name, dataframe in name_to_dataframe.items():
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
                        
        print_elapsed_time(start, prefix, name_to_dataframe)
        
    print("-------------------------------- Finishing Program --------------------------------")
