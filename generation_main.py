import time
from utils_generation.parser import getArgs
from utils_generation.load_utils import load_model, put_model_on_device, load_tokenizer, load_datasets
from utils_generation.generation import calZeroAndHiddenStates

if __name__ == "__main__":
    print("---------------- Program Begin ----------------")
    start = time.time()
    print("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    # get args
    args = getArgs()

    # load model and tokenizer (put model on hardware accelearator if possible)
    print("-------- Setting up model and tokenizer --------")
    print(f"loading model: model name = {args.model} at cache_dir = {args.cache_dir}")
    model = load_model(mdl_name=args.model, cache_dir=args.cache_dir)
    
    print(f"finish loading model to memory. Now start loading to accelerator (gpu or mps). parallelize = {args.parallelize == True}")
    model = put_model_on_device(model, parallelize=args.parallelize, device = args.model_device)
    
    print(f"loading tokenizer for: model name = {args.model} at cache_dir = {args.cache_dir}")
    tokenizer = load_tokenizer(mdl_name=args.model, cache_dir=args.cache_dir)

    prefix_list = args.prefix
    for prefix in prefix_list:
        args.prefix = prefix
        # load datasets and save if possible
        frame_dict = load_datasets(args, tokenizer)

        # for each frame, calculate the zero-shot accuracy and generate the hidden states if needed
        # the zero-shot accuracy will be stored in records
        # the hidden states will be saved to directories
        calZeroAndHiddenStates(model, tokenizer, frame_dict, args)

        end = time.time()
        print("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        print("Consider prefix {}, {} datasets, {} samples in total, and took {} minutes. Generation: {}, Hidden States: {}".format(
            prefix, len(frame_dict), sum([len(w) for w in frame_dict.values()]), round(
                (end - start) / 60, 1),  args.cal_zeroshot == True, args.cal_hiddenstates == True
        ))
        print("---------------------------------------\n\n")
    print("---------------- Program Finish ----------------")
