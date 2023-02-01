import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


print(
    "------ Func: scatter_data ------\n## Input = (data, label, n_con, name, idx) ##\n "
    "   data: The array you want to plot, with shape (len(data), n_con).\n    label:"
    " The classification label, with shape (len(data),).\n    n_con: The dimension of"
    " each data point. When n_con >= 2, only plot the first two dimension. Otherwise"
    " will plot the hist.\n    name & idx: the name and prompt of that this array comes"
    " from. Used only in plot title.\n## Output ##\n    No output. Directly show the"
    " plot.\n"
)


def scatter_data(data, label, n_con, name, idx):
    sns.set(font_scale=1.4)
    if n_con != 1:
        plt.figure(figsize=(9, 9))
    else:
        plt.figure(figsize=(8, 6))
    for j in range(2):
        mask = label == j
        if n_con >= 2 or n_con == -1:
            x, y = data[:, 0][mask], data[:, 1][mask]
            plt.scatter(
                x,
                y,
                label="{}_lbl{}".format("{}_idx{}".format(name, idx), j),
                alpha=0.7,
                marker="o" if j == 0 else "^",
            )
            plt.xlabel("The first component")
            plt.ylabel("The second component")
        else:
            temptag = ["True - False", "False - True"]
            x = data[:, 0][mask]
            df = pd.DataFrame(columns=["value", "label"])
            df["value"] = data[:, 0]
            df["label"] = [temptag[w] for w in label]
            sns.histplot(data=df, x="value", hue="label", alpha=0.85, kde=True, bins=30)
            plt.xlabel(r"$p^+ - p^-$", fontsize=20)

            plt.ylabel("")
            plt.xticks(range(-1, 1, 0.4))
            plt.yticks([])

    plt.show()
