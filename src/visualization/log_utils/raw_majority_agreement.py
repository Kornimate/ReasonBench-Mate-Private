from pathlib import Path

import pandas as pd

from .log_metric_common import (
    a4_landscape_size,
    add_method_highlight_legend,
    bar_colors,
    benchmark_groups,
    benchmark_output,
    draw_metric_bars,
    ensure_matplotlib,
    metric_axis_label,
    metric_plot,
    subplot_grid_size,
)


def write(df: pd.DataFrame, output_dir: Path) -> None:
    output = metric_plot(output_dir, "raw_majority_agreement")
    if df.empty or "method" not in df.columns:
        return
    label_columns = [column for column in ("method",) if column in df.columns]
    group_columns = [column for column in ("benchmark", *label_columns) if column in df.columns]
    plot_df = (
        df.dropna(subset=["majority_agreement"])
        .groupby(group_columns, dropna=False)["majority_agreement"]
        .mean()
        .reset_index()
    )
    if plot_df.empty or "benchmark" not in plot_df.columns:
        return
    plt = ensure_matplotlib()
    groups = benchmark_groups(plot_df)
    rows, columns = subplot_grid_size(len(groups))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False)
    for axis, (benchmark, benchmark_df) in zip(axes.flat, groups):
        labels = [
            "\n".join(str(getattr(row, column)) for column in label_columns if hasattr(row, column))
            for row in benchmark_df.itertuples()
        ]
        draw_metric_bars(axis, labels, benchmark_df["majority_agreement"], bar_colors(benchmark_df))
        axis.set_title(str(benchmark), fontsize=9)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(metric_axis_label("majority_agreement"), fontsize=8)
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        add_method_highlight_legend(axis, benchmark_df)
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    fig.suptitle("Raw Call Majority Agreement", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    for benchmark, benchmark_df in groups:
        labels = [
            "\n".join(str(getattr(row, column)) for column in label_columns if hasattr(row, column))
            for row in benchmark_df.itertuples()
        ]
        plt.figure(figsize=(max(8, len(labels) * 0.55), 5))
        draw_metric_bars(plt.gca(), labels, benchmark_df["majority_agreement"], bar_colors(benchmark_df))
        add_method_highlight_legend(plt.gca(), benchmark_df)
        plt.title(f"Raw Call Majority Agreement - {benchmark}")
        plt.ylabel(metric_axis_label("majority_agreement"))
        plt.ylim(0, 1.05)
        plt.tight_layout()
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(benchmark_path, dpi=180)
        plt.close()
