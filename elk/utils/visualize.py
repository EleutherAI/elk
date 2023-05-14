from pathlib import Path

import pandas as pd
import plotly.express as px


def generate_heatmap(data, model, layer, viz_dir: Path):
    pivot = pd.pivot_table(
        data, values="auroc_estimate", index="eval_dataset", columns="train_dataset"
    )

    fig = px.imshow(pivot, color_continuous_scale="Viridis", text_auto=True)

    fig.update_layout(
        xaxis_title="Train Dataset",
        yaxis_title="Transfer Dataset",
        title=f"AUROC Score Heatmap: {model} | Layer {layer}",
    )

    viz_dir = viz_dir / model
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig.write_image(viz_dir / f"{layer}.png")


def reduce_model_results(model_path):
    df = pd.DataFrame()

    for train_dir in model_path.iterdir():
        file = train_dir / "eval.csv"
        eval_df = pd.read_csv(file)
        eval_df["eval_dataset"] = train_dir.name
        eval_df["train_dataset"] = train_dir.name
        df = pd.concat([df, eval_df], ignore_index=True)
        transfer_dir = train_dir / "transfer"
        for eval_ds_dir in transfer_dir.iterdir():
            eval_file_path = eval_ds_dir / "eval.csv"
            eval_df = pd.read_csv(eval_file_path)
            eval_df["eval_dataset"] = eval_ds_dir.name
            eval_df["train_dataset"] = train_dir.name
            df = pd.concat([df, eval_df], ignore_index=True)

    return df


def render_model_results(root_dir, model):
    viz_dir = root_dir / "visualizations"
    print(f"Saving sweep visualizations to \033[1m{viz_dir}\033[0m")

    df = reduce_model_results(root_dir / model)
    layer_min, layer_max = df["layer"].min(), df["layer"].max()
    for layer in range(layer_min, layer_max + 1):
        df = df[df["ensembling"] == "full"]
        df = df[df["layer"] == layer]

        generate_heatmap(data=df, model=model, layer=layer, viz_dir=viz_dir)
