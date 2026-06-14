import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.visualization.plot_style import apply_plot_area_style, benchmark_label, method_color, method_label

from .log_metric_common import a4_landscape_size, benchmark_output, ensure_matplotlib, metric_plot, subplot_grid_size


def method_solved_rate_colors(matrix: pd.DataFrame) -> Any:
    """Return an RGBA image where each row uses that method's color."""
    import numpy as np
    from matplotlib.colors import to_rgba

    unsolved = np.array(to_rgba("#f3f6fb"))
    missing = np.array(to_rgba("#eef1f5"))
    values = matrix.values.astype(float)
    colors = np.zeros((*values.shape, 4), dtype=float)

    for row_index, method in enumerate(matrix.index):
        solved = np.array(to_rgba(method_color(method)))
        for column_index, value in enumerate(values[row_index]):
            if np.isnan(value):
                colors[row_index, column_index] = missing
                continue
            weight = min(1.0, max(0.0, float(value)))
            colors[row_index, column_index] = (1.0 - weight) * unsolved + weight * solved
    return colors


def write(df: pd.DataFrame, output_dir: Path) -> None:
    output = metric_plot(output_dir, "methods_heatmap")
    if df.empty or not {"benchmark", "method", "sample", "solved"}.issubset(df.columns):
        return
    plt = ensure_matplotlib()
    benchmarks = sorted(df["benchmark"].dropna().unique())
    matrices = [(benchmark, methods_heatmap_matrix(df, benchmark)) for benchmark in benchmarks]
    matrices = [(benchmark, matrix) for benchmark, matrix in matrices if not matrix.empty]
    if not matrices:
        return

    rows, columns = subplot_grid_size(len(matrices))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False, constrained_layout=True)
    for axis, (benchmark, matrix) in zip(axes.flat, matrices):
        draw_methods_heatmap(axis, matrix, benchmark_label(benchmark), show_labels=False)
    for axis in list(axes.flat)[len(matrices):]:
        axis.axis("off")
    fig.suptitle("Methods solved heatmap\nCells blend from pale unsolved to each method's color when solved", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    for benchmark, matrix in matrices:
        fig_width = min(15, max(9, len(matrix.columns) * 0.16))
        fig_height = min(8, max(4.8, len(matrix.index) * 0.42 + 1.2))
        fig, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        draw_methods_heatmap(axis, matrix, f"Methods solved heatmap - {benchmark_label(benchmark)}")
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(benchmark_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def methods_heatmap_matrix(df: pd.DataFrame, benchmark: Any) -> pd.DataFrame:
    benchmark_df = df[df["benchmark"].eq(benchmark)].dropna(subset=["solved"])
    if benchmark_df.empty:
        return pd.DataFrame()
    benchmark_df = benchmark_df.copy()
    benchmark_df["_source_order"] = range(len(benchmark_df))
    run_columns = [column for column in ("method", "split", "repeat") if column in benchmark_df.columns]
    benchmark_df = benchmark_df.sort_values([*run_columns, "_source_order"])
    benchmark_df["evaluation_index"] = benchmark_df.groupby(run_columns, dropna=False).cumcount() + 1
    plot_df = (
        benchmark_df.groupby(["method", "evaluation_index"], dropna=False)["solved"]
        .mean()
        .reset_index()
        .pivot(index="method", columns="evaluation_index", values="solved")
    )
    method_order = sorted(plot_df.index, key=lambda value: method_label(value))
    sample_order = sorted(plot_df.columns)
    return plot_df.reindex(index=method_order, columns=sample_order)


def draw_methods_heatmap(axis, matrix: pd.DataFrame, title: str, show_labels: bool = True):
    image = axis.imshow(method_solved_rate_colors(matrix), aspect="auto", interpolation="nearest")
    axis.set_title(title, fontsize=9 if show_labels else 8)

    column_count = len(matrix.columns)
    if column_count <= 20:
        x_positions = list(range(column_count))
    else:
        step = math.ceil(column_count / (12 if show_labels else 6))
        x_positions = list(range(0, column_count, step))
    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(matrix.columns[position]) for position in x_positions], rotation=45, ha="right", fontsize=7 if show_labels else 5)

    y_positions = list(range(len(matrix.index)))
    axis.set_yticks(y_positions)
    axis.set_yticklabels([method_label(method) for method in matrix.index], fontsize=8 if show_labels else 6)
    for tick, method in zip(axis.get_yticklabels(), matrix.index):
        tick.set_color(method_color(method))
        tick.set_fontweight("semibold")
    axis.tick_params(which="both", bottom=True, left=True)
    apply_plot_area_style(axis)

    if show_labels:
        axis.set_xlabel("Evaluated instance", fontsize=8)
        axis.set_ylabel("Method", fontsize=8)
        axis.text(
            1.0,
            -0.16,
            "Pale = not solved; row color = solved",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
        )

    row_count = len(matrix.index)
    if row_count <= 20 and column_count <= 60:
        axis.set_xticks([index - 0.5 for index in range(1, len(matrix.columns))], minor=True)
        axis.set_yticks([index - 0.5 for index in range(1, row_count)], minor=True)
        axis.grid(which="minor", axis="both", color="white", linewidth=0.6)
        axis.tick_params(which="minor", bottom=False, left=False)
    return image
