import time
from elk.utils_generation.parser import get_args
from elk.utils_generation.load_utils import load_model, load_datasets
from elk.utils_generation.generation import cal_zero_and_hidden_states


if __name__ == "__main__":
    print("---------------- Program Begin ----------------")
    start = time.time()
    print("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    # get args
    args = get_args()

    # load models, stored in GPU
    model, tokenizer = load_model(
        mdl_name=args.model, cache_dir=args.cache_dir, parallelize=args.parallelize
    )

    prefix_list = args.prefix
    for prefix in prefix_list:
        args.prefix = prefix
        # load datasets and save if possible
        frame_dict = load_datasets(args, tokenizer)

        # for each frame, calculate the zero-shot accuracy and generate the hidden
        # states if needed the zero-shot accuracy will be stored in records
        # the hidden states will be saved to directories
        cal_zero_and_hidden_states(model, tokenizer, frame_dict, args)

        end = time.time()
        print("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        print(
            "Consider prefix {}, {} datasets, {} samples in total, and took {} minutes."
            " Generation: {}, Hidden States: {}".format(
                prefix,
                len(frame_dict),
                sum([len(w) for w in frame_dict.values()]),
                round((end - start) / 60, 1),
                args.cal_zeroshot is True,
                args.cal_hiddenstates is True,
            )
        )
        print("---------------------------------------\n\n")
    print("---------------- Program Finish ----------------")
