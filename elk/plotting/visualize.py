from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

import elk.plotting.utils as utils
from elk.utils.constants import BURNS_DATASETS


class SweepByDsMultiplot:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def render(
        self,
        sweep: SweepVisualization,
        with_transfer=False,
        ensembles=["full", "partial", "none"],
        write=False,
    ) -> go.Figure:
        df = sweep.df
        unique_datasets = df["eval_dataset"].unique()
        run_names = df["run_name"].unique()
        num_datasets = len(unique_datasets)
        num_rows = (num_datasets + 2) // 3

        fig = make_subplots(
            rows=num_rows, cols=3, subplot_titles=unique_datasets, shared_yaxes=True
        )

        combos = list(itertools.product(run_names, ensembles))

        color_map = {
            run_name: color for run_name, color in zip(combos, qualitative.Plotly)
        }

        for run_name, ensemble in combos:
            run_data = df[df["run_name"] == run_name]
            ensemble_data: pd.DataFrame = run_data[run_data["ensembling"] == ensemble]
            if with_transfer:  # TODO write tests
                ensemble_data = ensemble_data.groupby(
                    ["eval_dataset", "layer", "run_name", "ensembling"], as_index=False
                ).agg({"auroc_estimate": "mean"})
            else:
                ensemble_data = ensemble_data[
                    ensemble_data["eval_dataset"] == ensemble_data["train_dataset"]
                ]
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
        if write:
            fig.write_image(
                file=sweep.path / f"{self.model_name}-line-ds-multiplot.png",
                scale=2,
            )
            fig.write_html(
                file=sweep.path / f"{self.model_name}-line-ds-multiplot.html"
            )

        return fig


class TransferEvalHeatmap:
    def __init__(
        self, layer: int, score_type: str = "auroc_estimate", ensembling: str = "full"
    ):
        self.layer = layer
        self.score_type = score_type
        self.ensembling = ensembling

    def render(self, df: pd.DataFrame, write=False) -> go.Figure:
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

    def render(self, df: pd.DataFrame) -> go.Figure:
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
class ModelVisualization:
    df: pd.DataFrame
    sweep_name: str
    model_name: str
    is_transfer: bool

    @classmethod
    def collect(cls, model_path: Path, sweep_name: str) -> ModelVisualization:
        df = pd.DataFrame()
        model_name = model_path.name

        is_transfer = False
        for train_dir in model_path.iterdir():
            eval_df = cls._read_eval_csv(train_dir, train_dir.name, train_dir.name)
            df = pd.concat([df, eval_df], ignore_index=True)
            transfer_dir = train_dir / "transfer"
            if transfer_dir.exists():
                is_transfer = True
                for eval_ds_dir in transfer_dir.iterdir():
                    eval_df = cls._read_eval_csv(
                        eval_ds_dir, eval_ds_dir.name, train_dir.name
                    )
                    df = pd.concat([df, eval_df], ignore_index=True)

        df["model_name"] = model_name
        df["run_name"] = sweep_name

        return cls(df, sweep_name, model_name, is_transfer)

    def render_and_save(
        self,
        sweep: SweepVisualization,
        dataset_names: list[str] = BURNS_DATASETS,
        score_type="auroc_estimate",
        ensembling="full",
    ) -> None:
        df = self.df
        model_name = self.model_name
        layer_min, layer_max = df["layer"].min(), df["layer"].max()
        model_path = sweep.path / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        if self.is_transfer:
            for layer in range(layer_min, layer_max + 1):
                filtered = df[(df["layer"] == layer) & (df["ensembling"] == ensembling)]
                fig = TransferEvalHeatmap(
                    layer, score_type=score_type, ensembling=ensembling
                ).render(filtered)
                fig.write_image(file=model_path / f"{layer}.png")
        fig = TransferEvalTrend(dataset_names).render(df)
        fig.write_image(file=model_path / "transfer_eval_trend.png")

    @staticmethod
    def _read_eval_csv(path, eval_dataset, train_dataset):
        file = path / "eval.csv"
        eval_df = pd.read_csv(file)
        eval_df["eval_dataset"] = eval_dataset
        eval_df["train_dataset"] = train_dataset
        return eval_df


@dataclass
class SweepVisualization:
    name: str
    df: pd.DataFrame
    path: Path
    datasets: list[str]
    models: dict[str, ModelVisualization]

    def model_names(self):
        return list(self.models.keys())

    @staticmethod
    def _get_model_paths(sweep_path: Path) -> list[Path]:
        folders = []
        for model_repo in sweep_path.iterdir():
            if not model_repo.is_dir():
                raise Exception(f"expected {model_repo} to be a directory")
            if model_repo.name.startswith("gpt2"):
                folders += [model_repo]
            else:
                folders += [p for p in model_repo.iterdir() if p.is_dir()]
        return folders

    @classmethod
    def collect(cls, sweep_path: Path) -> SweepVisualization:
        sweep_name = sweep_path.parts[-1]
        sweep_viz_path = sweep_path / "viz"
        if sweep_viz_path.exists():
            raise Exception("This sweep has already been visualized.")
        sweep_viz_path.mkdir(parents=True, exist_ok=True)

        model_paths = cls._get_model_paths(sweep_path)
        models = {
            model_path.name: ModelVisualization.collect(model_path, sweep_name)
            for model_path in model_paths
        }
        df = pd.concat([model.df for model in models.values()], ignore_index=True)
        datasets = list(df["eval_dataset"].unique())
        return cls(sweep_name, df, sweep_viz_path, datasets, models)

    def render_and_save(self):
        for model in self.models.values():
            model.render_and_save(self)
        self.render_table(write=True)
        self.render_multiplots(write=True)

    def render_multiplots(self, write=False):
        return [
            SweepByDsMultiplot(model).render(self, write=write, with_transfer=False)
            for model in self.models
        ]

    def render_table(
        self, layer=-5, score_type="auroc_estimate", display=True, write=False
    ):
        df = self.df
        layer_by_model = (df.groupby("model_name")["layer"].max() + layer).clip(lower=0)
        df_selected_layer = pd.DataFrame()
        for model, layer in layer_by_model.items():
            record = df[(df["model_name"] == model) & (df["layer"] == layer)]
            df_selected_layer = pd.concat([df_selected_layer, record])
        pivot_table = df_selected_layer.pivot_table(
            index="run_name", columns="model_name", values=score_type
        )
        if display:
            utils.display_table(pivot_table)
        if write:
            pivot_table.to_csv(f"score_table_{score_type}.csv")
        return pivot_table


def visualize_sweep(sweep_path: Path):
    SweepVisualization.collect(sweep_path).render_and_save()
