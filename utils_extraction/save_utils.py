import numpy as np
import os
import time 

def save_params(args, name, coef, intercept):
    path = os.path.join(args.save_dir, "params")
    np.save(os.path.join(path, "coef_{}.npy".format(name)), coef)
    np.save(os.path.join(path, "intercept_{}.npy".format(name)), intercept)

def save_df_to_csv(args, df, prefix, str = ""):
    dir = os.path.join(args.save_dir, "{}_{}_{}.csv".format(args.model, prefix, args.seed))
    df.to_csv(dir, index = False)
    print("{} Saving to {} at {}".format(str, dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

