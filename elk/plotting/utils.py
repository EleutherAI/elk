import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table


def set_subplot_title_font_size(fig, font_size=14, font_family="Arial"):
    fig.update_annotations(
        font=dict(
            family=font_family,
            size=font_size,
        )
    )
    return fig


def set_legend_font_size(fig, font_size=8, font_family="Arial"):
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(
                family=font_family,
                size=font_size,
            )
        ),
        yaxis=dict(
            tickfont=dict(
                family=font_family,
                size=font_size,
            )
        ),
        legend=dict(
            font=dict(
                family=font_family,
                size=font_size,
            )
        ),
    )
    return fig


def display_table(pivot_table):
    console = Console()

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)

    # Add columns
    table.add_column("Run Name")
    for column in pivot_table.columns:
        table.add_column(str(column))

    # Add rows
    for index, row in pivot_table.iterrows():
        table.add_row(str(index), *[str(value) for value in row])

    console.print(table)


def restructure_to_sweep(
    elk_reporters: Path, data_path, new_name: str
):  # usually /elk-reporters/*
    for model_repo_path in elk_reporters.iterdir():
        for model_path in model_repo_path.iterdir():
            for dataset_path in model_path.iterdir():
                for run_path in dataset_path.iterdir():
                    new_path = (
                        data_path
                        / new_name
                        / run_path.name
                        / model_repo_path.name
                        / model_path.name
                        / dataset_path.name
                    )
                    if not new_path.exists():
                        new_path.mkdir(parents=True)
                    for file in run_path.iterdir():
                        if file.is_file():
                            shutil.copy(file, new_path / file.name)
                        else:
                            shutil.copytree(file, new_path / file.name)
