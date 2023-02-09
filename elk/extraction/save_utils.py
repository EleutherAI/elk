import numpy as np
import pandas as pd
import time


def getDir(dataset_name_w_num, args):
    d = "{}_{}_{}_{}".format(
        args.model, dataset_name_w_num, args.prefix, args.token_loc
    )
    if args.tag != "":
        d += "_{}".format(args.tag)

    return args.save_base_dir / d


def saveFrame(frame_dict, args):
    args.save_base_dir.mkdir(parents=True, exist_ok=True)

    for key, frame in frame_dict.items():

        directory = getDir(key, args)
        directory.mkdir(parents=True, exist_ok=True)

        frame.to_csv(directory / "frame.csv", index=False)

    print("Successfully saving datasets to each directory.")


def saveArray(array_list, typ_list, key, args):
    directory = getDir(key, args)
    directory.mkdir(parents=True, exist_ok=True)

    # hidden states is num_data * layers * dim
    # logits is num_data * vocab_size
    for (typ, array) in zip(typ_list, array_list):
        if args.layers:
            for idx in args.layers:
                np.save(
                    directory / f"{typ}_{idx}.npy",
                    array[:, idx, :],
                )
        else:
            np.save(directory / f"{typ}.npy", array)


def save_records_to_csv(records, args):
    f = args.save_base_dir / f"{args.save_csv_name}.csv"
    if not f.exists():
        csv = pd.DataFrame(
            columns=[
                "time",
                "model",
                "dataset",
                "prompt_idx",
                "num_data",
                "population",
                "prefix",
                "cal_zeroshot",
                "log_probs",
                "calibrated",
                "tag",
            ]
        )
    else:
        csv = pd.read_csv(f)

    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for dic in records:
        dic["time"] = t
        spliter = dic["dataset"].split("_")
        dic["dataset"], dic["prompt_idx"] = spliter[0], int(spliter[2][6:])
    csv = csv.append(records, ignore_index=True)

    csv.to_csv(f, index=False)

    print("Successfully saved {} items in records to {}".format(len(records), f))
