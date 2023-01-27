import time
from utils_generation.parser import getArgs
from utils_generation.load_utils import load_model, put_model_on_device, load_tokenizer, load_datasets
from utils_generation.generation import create_records, create_hiddenstates

if __name__ == "__main__":
    print("\n\n-------------------------------- Starting Program --------------------------------\n\n")
    start = time.time()
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # get args
    args = getArgs()

    # load model and tokenizer (put model on hardware accelearator if possible)
    print("\n\n--------------------------------  Setting up model and tokenizer --------------------------------\n\n")
    print(f"loading model: model name = {args.model} at cache_dir = {args.cache_dir}")
    model = load_model(mdl_name=args.model, cache_dir=args.cache_dir)
    
    print(f"finish loading model to memory. Now start loading to accelerator (gpu or mps). parallelize = {args.parallelize is True}")
    model = put_model_on_device(model, parallelize=args.parallelize, device = args.model_device)
    
    print(f"loading tokenizer for: model name = {args.model} at cache_dir = {args.cache_dir}")
    tokenizer = load_tokenizer(mdl_name=args.model, cache_dir=args.cache_dir)

    print("\n\n-------------------------------- Loading datasets and calculating hidden states --------------------------------\n\n")
    all_prefixes = args.prefix
    for prefix in all_prefixes:
        args.prefix = prefix
        # load datasets and save if possible
        name_to_dataframe = load_datasets(args, tokenizer)

        # For each frame, generate the hidden states and save to directories
        print("-------- Generating hidden states --------")
        create_hiddenstates(model, tokenizer, name_to_dataframe, args)
        create_records(model, tokenizer, name_to_dataframe, args)
        
        total_samples = sum([len(dataframe) for dataframe in name_to_dataframe.values()])
        end = time.time()
        elapsed_minutes = round((end - start) / 60, 1)
        print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        print(f"Prefix used: {prefix}, applied to {len(name_to_dataframe)} datasets, {total_samples} samples in total, and took {elapsed_minutes} minutes.")
        print("\n\n---------------------------------------\n\n")
    
    print("-------------------------------- Finishing Program --------------------------------")
