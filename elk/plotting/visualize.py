from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

import elk.plotting.utils as utils


@dataclass
class Plot:
    sweeps: list[Path] = field(default_factory=list)

    def execute(self):
        sweeps_path = Path.home() / "elk-reporters" / "sweeps"
        # in sweeps_path find the most recent sweep
        sweep = max(sweeps_path.iterdir(), key=os.path.getctime)
        if self.sweeps:
            sweep = self.sweeps[0]
        if len(self.sweeps) > 1:
            print(
                f"""{len(self.sweeps)} paths specified.
                Only one sweep is supported at this time."""
            )
        else:
            visualize_sweep(sweep)  # TODO support more than one sweep


VIZ_PATH = Path(os.getcwd()) / "viz"
ALL_DS_NAMES = [
    "super_glue:rte",
    "super_glue:boolq",
    "dbpedia_14",
    "piqa",
    "amazon_polarity",
    "glue:qnli",
    "ag_news",
    "imdb",
    "super_glue:copa",
]


METHODS = [
    ("ccs-ar", "none"),
    ("vinc-std-norm-ar", "none"),
    ("vinc-std-norm-ar", "full"),
    ("crctpc", "none"),
]


class SweepByDsMultiplot:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load() -> SweepByDsMultiplot:
        pass

    def validate(self, df) -> bool:
        # validate only one
        pass

    def render(self, sweep: Sweep):
        df = sweep.df
        unique_datasets = df["eval_dataset"].unique()
        run_names = df["run_name"].unique()
        ensembles = ["full", "none"]
        num_datasets = len(unique_datasets)
        num_rows = (num_datasets + 2) // 3

        fig = make_subplots(
            rows=num_rows, cols=3, subplot_titles=unique_datasets, shared_yaxes=True
        )

        # combos = METHODS
        combos = list(itertools.product(run_names, ensembles))

        color_map = {
            run_name: color for run_name, color in zip(combos, qualitative.Plotly)
        }

        for run_name, ensemble in combos:
            run_data = df[df["run_name"] == run_name]
            ensemble_data = run_data[run_data["ensembling"] == ensemble]

            for i, dataset_name in enumerate(unique_datasets, start=1):
                dataset_data = ensemble_data[
                    ensemble_data["eval_dataset"] == dataset_name
                ]
                dataset_data = dataset_data.sort_values(by="layer")
                row, col = (i - 1) // 3 + 1, (i - 1) % 3 + 1
                fig.add_trace(
                    go.Scatter(
                        x=dataset_data["layer"],
                        y=dataset_data["auroc_estimate"],
                        mode="lines",
                        name=f"{run_name}:{ensemble}",
                        showlegend=False
                        if dataset_name != unique_datasets[0]
                        else True,
                        line=dict(color=color_map[(run_name, ensemble)]),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            xaxis_title="Layer",
            yaxis_title="AUROC Score",
            title=f"Auroc Score Trend: {self.model_name}",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(t=100),
        )

        for i in range(1, num_datasets + 1):
            fig.update_xaxes(
                title_text="Layer", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1
            )

        for i in range(1, num_rows + 1):
            fig.update_yaxes(title_text="AUROC Score", row=i, col=1)

        fig = utils.set_subplot_title_font_size(fig, font_size=8)
        fig = utils.set_legend_font_size(fig, font_size=8)
        fig.write_image(
            file=VIZ_PATH / sweep.name / f"{self.model_name}-line-ds-multiplot.png",
            scale=2,
        )
        fig.write_html(
            file=VIZ_PATH / sweep.name / f"{self.model_name}-line-ds-multiplot.html"
        )

        return fig


class TransferEvalHeatmap:
    def __init__(
        self, layer: int, score_type: str = "auroc_estimate"
    ):  # TODO make enum
        self.layer = layer
        self.score_type = score_type

    def load() -> TransferEvalHeatmap:
        pass

    def validate(self, df) -> bool:
        # validate only one
        pass

    def generate(self, df: pd.DataFrame) -> go.Figure:
        """
        Generate a heatmap for dataset of a model.
        """
        model_name = df["eval_dataset"].iloc[0]  # infer model name
        # TODO: validate
        pivot = pd.pivot_table(
            df, values=self.score_type, index="eval_dataset", columns="train_dataset"
        )

        fig = px.imshow(pivot, color_continuous_scale="Viridis", text_auto=True)

        fig.update_layout(
            xaxis_title="Train Dataset",
            yaxis_title="Transfer Dataset",
            title=f"AUROC Score Heatmap: {model_name} | Layer {self.layer}",
        )

        return fig


class TransferEvalTrend:
    def __init__(self, dataset_names, score_type: str = "auroc_estimate"):
        self.dataset_names = dataset_names
        self.score_type = score_type

    def load() -> TransferEvalTrend:
        pass

    def validate(self, df) -> bool:
        # validate only one
        pass

    def generate(self, df: pd.DataFrame) -> go.Figure:
        # TODO should I filter out the non-transfer dataset?
        model_name = df["eval_dataset"].iloc[0]  # infer model name
        df = self._filter_transfer_datasets(df, self.dataset_names)
        pivot = pd.pivot_table(
            df, values=self.score_type, index="layer", columns="eval_dataset"
        )

        fig = px.line(pivot, color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(
            xaxis_title="Layer",
            yaxis_title="AUROC Score",
            title=f"AUROC Score Trend: {model_name}",
        )

        # add line representing average of all datasets
        avg = pivot.mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=avg.index,
                y=avg.values,
                name="average",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        return fig

    def _filter_transfer_datasets(self, df, dataset_names):
        df = df[df["eval_dataset"].isin(dataset_names)]
        df = df[df["train_dataset"].isin(dataset_names)]
        return df


@dataclass
class Model:
    df: pd.DataFrame
    sweep_name: str
    model_name: str
    is_transfer: bool

    @classmethod
    def collect(cls, model_path: Path, sweep_name: str) -> Model:
        df = pd.DataFrame()
        model_name = model_path.name

        def handle_csv(dir, eval_dataset, train_dataset):
            file = dir / "eval.csv"
            eval_df = pd.read_csv(file)
            eval_df["eval_dataset"] = eval_dataset
            eval_df["train_dataset"] = train_dataset
            return eval_df

        is_transfer = False
        for train_dir in model_path.iterdir():
            eval_df = handle_csv(train_dir, train_dir.name, train_dir.name)
            df = pd.concat([df, eval_df], ignore_index=True)
            transfer_dir = train_dir / "transfer"
            if transfer_dir.exists():
                is_transfer = True
                for eval_ds_dir in transfer_dir.iterdir():
                    eval_df = handle_csv(eval_ds_dir, eval_ds_dir.name, train_dir.name)
                    df = pd.concat([df, eval_df], ignore_index=True)

        df["model_name"] = model_name
        df["run_name"] = sweep_name

        return cls(df, sweep_name, model_name, is_transfer)

    def render(self, dataset_names: list[str] = ALL_DS_NAMES):
        df = self.df
        model_name = self.model_name
        is_transfer = self.is_transfer
        sweep_name = self.sweep_name
        layer_min, layer_max = df["layer"].min(), df["layer"].max()
        if not (VIZ_PATH / sweep_name).exists():
            (VIZ_PATH / sweep_name).mkdir()
        if not (VIZ_PATH / sweep_name / f"{model_name}").exists():
            (VIZ_PATH / sweep_name / f"{model_name}").mkdir()
        if is_transfer:
            for layer in range(layer_min, layer_max + 1):

                def filter_df(df, ensembling, layer):
                    df = df[df["ensembling"] == ensembling]
                    df = df[df["layer"] == layer]
                    return df

                filtered = filter_df(df, "full", layer)

                path = VIZ_PATH / sweep_name / f"{model_name}" / f"{layer}.png"
                if not path.parent.exists():
                    path.parent.mkdir()
                fig = TransferEvalHeatmap(layer).generate(filtered)
                fig.write_image(file=path)

        TransferEvalTrend(dataset_names).generate(df).write_image(
            file=VIZ_PATH / sweep_name / f"{model_name}" / "trend.png"
        )


def reduce_model_results(model_path: Path, run_name: str) -> Model:
    df = pd.DataFrame()
    model_name = model_path.name

    def handle_csv(dir, eval_dataset, train_dataset):
        file = dir / "eval.csv"
        eval_df = pd.read_csv(file)
        eval_df["eval_dataset"] = eval_dataset
        eval_df["train_dataset"] = train_dataset
        return eval_df

    is_transfer = False
    for train_dir in model_path.iterdir():
        eval_df = handle_csv(train_dir, train_dir.name, train_dir.name)
        df = pd.concat([df, eval_df], ignore_index=True)
        transfer_dir = train_dir / "transfer"
        if transfer_dir.exists():
            is_transfer = True
            for eval_ds_dir in transfer_dir.iterdir():
                eval_df = handle_csv(eval_ds_dir, eval_ds_dir.name, train_dir.name)
                df = pd.concat([df, eval_df], ignore_index=True)

    df["model_name"] = model_name
    df["run_name"] = run_name

    return Model(df, run_name, model_name, is_transfer)


def render_model_results(res: Model, dataset_names: list[str] = ALL_DS_NAMES):
    df = res.df
    model_name = res.model_name
    is_transfer = res.is_transfer
    sweep_name = res.sweep_name
    layer_min, layer_max = df["layer"].min(), df["layer"].max()
    if not (VIZ_PATH / sweep_name).exists():
        (VIZ_PATH / sweep_name).mkdir()
    if not (VIZ_PATH / sweep_name / f"{model_name}").exists():
        (VIZ_PATH / sweep_name / f"{model_name}").mkdir()
    if is_transfer:
        for layer in range(layer_min, layer_max + 1):

            def filter_df(df, ensembling, layer):
                df = df[df["ensembling"] == ensembling]
                df = df[df["layer"] == layer]
                return df

            filtered = filter_df(df, "full", layer)

            path = VIZ_PATH / sweep_name / f"{model_name}" / f"{layer}.png"
            if not path.parent.exists():
                path.parent.mkdir()
            fig = TransferEvalHeatmap(layer).generate(filtered)
            fig.write_image(file=path)

    TransferEvalTrend(dataset_names).generate(df).write_image(
        file=VIZ_PATH / sweep_name / f"{model_name}" / "trend.png"
    )


# the following function does too many things.
# it can be split up into:
# function that takes a sweep/run and returns a dataframe
# function that takes df runs / run / model (with or without transfer) and renders it


@dataclass
class Sweep:
    name: str
    df: pd.DataFrame
    path: Path
    datasets: list[str]
    models: list[str]

    @classmethod
    def collect(cls, sweep: Path) -> Sweep:
        sweep_name = sweep.parts[-1]
        sweep_viz_path = VIZ_PATH / sweep_name

        # TODO refactor out
        if not VIZ_PATH.exists():
            VIZ_PATH.mkdir()
        if not sweep_viz_path.exists():
            sweep_viz_path.mkdir()

        model_paths = get_model_paths(sweep)
        df = pd.DataFrame()
        for model_path in model_paths:
            model_res = Model.collect(model_path, sweep_name)
            render_model_results(model_res)
            df = pd.concat([df, model_res.df], ignore_index=True)
        # make new dataclass here that
        datasets = list(df["eval_dataset"].unique())
        models = list(df["model_name"].unique())
        return cls(sweep_name, df, sweep_viz_path, datasets, models)

    def generate_multiplots(self):
        return [SweepByDsMultiplot(model).render(self) for model in self.models]

    def generate_table(
        self, layer=-5, score_type="auroc_estimate", print=True, write=False
    ):
        df = self.df

        layer_by_model = (df.groupby("model_name")["layer"].max() + layer).clip(lower=0)

        # Create an empty DataFrame to store the selected records
        df_selected_layer = pd.DataFrame()

        # For each model, select the record corresponding to max layer - 5
        for model, layer in layer_by_model.items():
            record = df[(df["model_name"] == model) & (df["layer"] == layer)]
            df_selected_layer = pd.concat([df_selected_layer, record])

        # Generate the pivot table
        pivot_table = df_selected_layer.pivot_table(
            index="run_name", columns="model_name", values=score_type
        )

        if print:
            utils.display_table(pivot_table)
        if write:
            pivot_table.to_csv(f"score_table_{score_type}.csv")

        return pivot_table


def get_model_paths(sweep_path: Path) -> list[Path]:
    # TODO write test
    # run / model_repo / model / dataset / eval.csv
    folders = []
    for model_repo in sweep_path.iterdir():
        if not model_repo.is_dir():
            raise Exception("expected model repo to be a directory")
        if model_repo.name.startswith("gpt2"):
            folders += [model_repo]
        else:
            folders += [p for p in model_repo.iterdir() if p.is_dir()]
    return folders


def visualize_sweep(sweep_path: Path):
    sweep = Sweep.collect(sweep_path)
    # special data filtering
    # NOTE TRANSFER: select only data where eval_dataset == train_dataset
    # all_data = all_data[all_data["eval_dataset"] == all_data["train_dataset"]]
    # NOTE ensembling: select only data where ensembling == full
    # all_data = all_data[all_data["ensembling"] == "full"]
    # NOTE modify all the auroc to be the max of auroc and (1-auroc)
    # NOTE this should really be handled data side
    # all_data["auroc_estimate"] = all_data["auroc_estimate"].apply(
    #     lambda x: max(x, 1 - x)
    # )
    sweep.generate_table(write=True)
    sweep.generate_multiplots()


if __name__ == "__main__":
    root = Path(os.getcwd())
    if not VIZ_PATH.exists():
        VIZ_PATH.mkdir()
    utils.restructure_to_sweep(root / "elk-reporters", root / "data", "platt-4")
    visualize_sweep(root / "data" / "platt-4")
