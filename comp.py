import pandas as pd

# import df from csv from paths
paths = [
    "/home/jon/elk-reporters/gpt2/imdb/optimistic-lamport/eval.csv",
    "/home/jon/elk-reporters/gpt2/imdb/hopeful-gould/eval.csv",
]

df = pd.read_csv(paths[0])
df_with_params = pd.read_csv(paths[1])

# get diff of all values that are floats
df_diff = df.select_dtypes(include=["float64"]) - df_with_params.select_dtypes(
    include=["float64"]
)
# reinsert the non-float columns at the front
df_diff = pd.concat([df.select_dtypes(include=["object", "int"]), df_diff], axis=1)

import rich

rich.print(df_diff)
rich.print(df_diff.describe())
