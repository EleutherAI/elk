import numpy as np
import os
import time 

def save_params(args, name, coef, intercept):
    path = os.path.join(args.save_dir, "params")
    np.save(os.path.join(path, f"coef_{name}.npy"), coef)
    np.save(os.path.join(path, f"intercept_{name}.npy"), intercept)

def save_df_to_csv(args, df, prefix, str = ""):
    dir = os.path.join(args.save_dir, f"{args.model}_{prefix}_{args.seed}.csv")
    df.to_csv(dir, index = False)
    print(f"{str} Saving to {dir} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

