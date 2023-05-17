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

    table.add_column("Run Name")
    for column in pivot_table.columns:
        table.add_column(str(column))

    for index, row in pivot_table.iterrows():
        table.add_row(str(index), *[str(value) for value in row])

    console.print(table)
