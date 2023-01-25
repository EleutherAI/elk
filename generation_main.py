import time
from utils_generation.parser import getArgs
from utils_generation.load_utils import load_model, put_model_on_device, load_tokenizer, load_datasets
from utils_generation.generation import create_dataset_zeroshot_hiddenstates_records
from utils_generation.save_utils import save_records_to_csv

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
    prefix_list = args.prefix
    for prefix in prefix_list:
        args.prefix = prefix
        # load datasets and save if possible
        name_to_dataframe = load_datasets(args, tokenizer)

        # for each frame, calculate the zero-shot accuracy and generate the hidden states if needed
        # the zero-shot accuracy will be stored in records
        # the hidden states will be saved to directories
        print("-------- zero-shot & generation --------")

        records = create_dataset_zeroshot_hiddenstates_records(model, tokenizer, name_to_dataframe, args)
        save_records_to_csv(records)

        end = time.time()
        print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        total_samples = sum([len(dataframe) for dataframe in name_to_dataframe.values()])
        elapsed_minutes = round((end - start) / 60, 1)
        print(f"""
            Consider prefix {prefix}, {len(name_to_dataframe)} datasets, {total_samples} samples in total, and took {elapsed_minutes} minutes. 
            Generation: {args.cal_zeroshot is True}, Hidden States: {args.cal_hiddenstates is True}
            """)
        print("\n\n---------------------------------------\n\n")
    print("-------------------------------- Finishing Program --------------------------------")
