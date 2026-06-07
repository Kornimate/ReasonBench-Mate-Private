import math
from pathlib import Path
from typing import Any

import pandas as pd

from .log_metric_common import a4_landscape_size, benchmark_output, ensure_matplotlib, metric_plot, subplot_grid_size


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
    last_image = None
    for axis, (benchmark, matrix) in zip(axes.flat, matrices):
        last_image = draw_methods_heatmap(axis, matrix, str(benchmark), show_labels=False)
    for axis in list(axes.flat)[len(matrices):]:
        axis.axis("off")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), ticks=[0, 1], shrink=0.72)
        cbar.set_ticklabels(["Not solved", "Solved"])
        cbar.set_label("Solved status (0/1)")
    fig.suptitle("Methods Solved Heatmap", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    for benchmark, matrix in matrices:
        fig, axis = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.7), max(6, len(matrix.index) * 0.16)))
        image = draw_methods_heatmap(axis, matrix, f"Methods Solved Heatmap - {benchmark}")
        cbar = fig.colorbar(image, ax=axis, label="Solved status (0/1)", ticks=[0, 1], orientation="vertical")
        cbar.set_ticklabels(["Not solved", "Solved"])
        fig.tight_layout()
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(benchmark_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def methods_heatmap_matrix(df: pd.DataFrame, benchmark: Any) -> pd.DataFrame:
    benchmark_df = df[df["benchmark"].eq(benchmark)].dropna(subset=["solved"])
    if benchmark_df.empty:
        return pd.DataFrame()
    benchmark_df = benchmark_df.copy()
    label_columns = [column for column in ("model", "method", "split", "repeat") if column in benchmark_df.columns]
    benchmark_df["run_label"] = benchmark_df.apply(lambda row: heatmap_run_label(row, label_columns), axis=1)
    plot_df = (
        benchmark_df.groupby(["sample", "run_label"], dropna=False)["solved"]
        .mean()
        .reset_index()
        .pivot(index="sample", columns="run_label", values="solved")
        .sort_index()
    )
    return plot_df.reindex(sorted(plot_df.columns), axis=1)


def heatmap_run_label(row: pd.Series, label_columns: list[str]) -> str:
    parts = []
    for column in label_columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        if column == "repeat":
            try:
                parts.append(f"r{int(value)}")
            except (TypeError, ValueError):
                parts.append(f"r{value}")
        else:
            parts.append(str(value))
    return " / ".join(parts) if parts else "unknown"


def draw_methods_heatmap(axis, matrix: pd.DataFrame, title: str, show_labels: bool = True):
    image = axis.imshow(matrix.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_title(title, fontsize=9 if show_labels else 8)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7 if show_labels else 5)

    row_count = len(matrix.index)
    if row_count <= 20:
        y_positions = list(range(row_count))
    else:
        step = math.ceil(row_count / 10)
        y_positions = list(range(0, row_count, step))
    axis.set_yticks(y_positions)
    axis.set_yticklabels([f"Run {int(matrix.index[position])}" for position in y_positions], fontsize=7 if show_labels else 5)
    axis.tick_params(which="both", bottom=True, left=True)

    if show_labels:
        axis.set_xlabel("Method", fontsize=8)
        axis.set_ylabel("Sample index (instance)", fontsize=8)

    if row_count <= 60 and len(matrix.columns) <= 20:
        axis.set_xticks([index - 0.5 for index in range(1, len(matrix.columns))], minor=True)
        axis.set_yticks([index - 0.5 for index in range(1, row_count)], minor=True)
        axis.grid(which="minor", axis="both", color="white", linewidth=0.4)
        axis.tick_params(which="minor", bottom=False, left=False)
    return image
