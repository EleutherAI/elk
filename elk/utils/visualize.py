import os
from pathlib import Path

import pandas as pd
import plotly.express as px


def generate_heatmap(data, model_name, layer, viz_folder_path):
    
    pivot = pd.pivot_table(
        data, values="auroc_estimate", index="eval_dataset", columns="train_dataset"
    )

    fig = px.imshow(pivot, color_continuous_scale="Viridis", text_auto=True)

    fig.update_layout(
        xaxis_title="Train Dataset",
        yaxis_title="Transfer Dataset",
        title=f"AUROC Score Heatmap: {model_name} | Layer {layer}",
    )

    model_viz_folder_path = os.path.join(viz_folder_path, model_name)
    if not os.path.exists(model_viz_folder_path):
        os.mkdir(model_viz_folder_path)
    fig.write_image(os.path.join(model_viz_folder_path, f"{layer}.png"))


def filter_df(df, layer):
    df = df[df["ensembling"] == "full"]  # filter df to only include full ensembling
    df = df[df["layer"] == layer]  # filter df to only include target layer
    return df


def render_model_results(model_path, visualizations_path):
    if not visualizations_path.exists():
        visualizations_path.mkdir()
    
    df = None

    def get_layer_min_max(model_path: Path):
        dir = model_path.iterdir().__next__()
        file = os.path.join(dir, "eval.csv")
        raw_eval_df = pd.read_csv(file)
        layer_min, layer_max = raw_eval_df["layer"].min(), raw_eval_df["layer"].max()
        return layer_min, layer_max

    layer_min, layer_max = get_layer_min_max(model_path)

    for layer in range(layer_min, layer_max + 1):
        for dir in model_path.iterdir():
            file = os.path.join(dir, "eval.csv")
            raw_eval_df = pd.read_csv(file)

            eval_df = filter_df(raw_eval_df, layer)
            eval_df["eval_dataset"] = dir.name
            eval_df["train_dataset"] = dir.name
            if df is None:
                df = eval_df  # first time
            else:
                df = pd.concat([df, eval_df], ignore_index=True)

            transfer_dir = Path(os.path.join(dir, "transfer"))
            for eval_ds_dir in transfer_dir.iterdir():
                eval_file_path = os.path.join(eval_ds_dir, "eval.csv")
                raw_eval_df = pd.read_csv(eval_file_path)
                eval_df = filter_df(raw_eval_df, layer)
                eval_df["eval_dataset"] = eval_ds_dir.name
                eval_df["train_dataset"] = dir.name
                df = pd.concat([df, eval_df], ignore_index=True)

        model_name = model_path.parts[-1]
        generate_heatmap(df, model_name, layer, visualizations_path)
