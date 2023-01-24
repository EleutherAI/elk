import time
from utils_generation.parser import getArgs
from utils_generation.load_utils import loadModel, loadDatasets
from utils_generation.generation import calZeroAndHiddenStates

if __name__ == "__main__":
    print("---------------- Program Begin ----------------")
    start = time.time()
    print("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    # get args
    args = getArgs()

    # load models, stored in GPU
    model, tokenizer = loadModel(
        mdl_name=args.model, cache_dir=args.cache_dir, parallelize=args.parallelize, device=args.model_device)

    prefix_list = args.prefix
    for prefix in prefix_list:
        args.prefix = prefix
        # load datasets and save if possible
        frame_dict = loadDatasets(args, tokenizer)

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
