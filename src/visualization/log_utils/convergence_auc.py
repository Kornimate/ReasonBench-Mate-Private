import json
from pathlib import Path
from statistics import mean

import pandas as pd

from .log_metric_common import (
    a4_landscape_size,
    benchmark_groups,
    benchmark_output,
    ensure_matplotlib,
    metric_plot,
    subplot_grid_size,
)


def write(df: pd.DataFrame, output_dir: Path) -> None:
    output = metric_plot(output_dir, "convergence_auc")
    if df.empty or "trajectory" not in df.columns:
        return
    plt = ensure_matplotlib()
    label_columns = [column for column in ("method",) if column in df.columns]
    if "benchmark" not in df.columns or not label_columns:
        return

    groups = benchmark_groups(df)
    rows, columns = subplot_grid_size(len(groups))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False)
    has_overview_curve = False
    for axis, (benchmark, benchmark_df) in zip(axes.flat, groups):
        has_curve = False
        for group_key, group in benchmark_df.groupby(label_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label = "/".join(str(value) for value in group_key)
            curves = [_loads_trajectory(value) for value in group["trajectory"]]
            curves = [curve for curve in curves if curve]
            if not curves:
                continue
            y = _mean_curve(curves)
            axis.plot(range(len(y)), y, marker="o", markersize=2, linewidth=1, label=label)
            has_curve = True
            has_overview_curve = True
        axis.set_title(str(benchmark), fontsize=9)
        axis.set_xlabel("logged step position (index)", fontsize=8)
        axis.set_ylabel("running best score", fontsize=8)
        axis.tick_params(labelsize=7)
        if has_curve:
            axis.legend(fontsize=5, loc="best")
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    if has_overview_curve:
        fig.suptitle("Logged Best-Score Convergence", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    plt.close(fig)

    for benchmark, benchmark_df in groups:
        plt.figure(figsize=(12, 6))
        has_curve = False
        for group_key, group in benchmark_df.groupby(label_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label = "/".join(str(value) for value in group_key)
            curves = [_loads_trajectory(value) for value in group["trajectory"]]
            curves = [curve for curve in curves if curve]
            if not curves:
                continue
            plt.plot(range(len(_mean_curve(curves))), _mean_curve(curves), marker="o", label=label)
            has_curve = True
        if not has_curve:
            plt.close()
            continue
        plt.title(f"Logged Best-Score Convergence - {benchmark}")
        plt.xlabel("Logged step position (index)")
        plt.ylabel("Running best logged score")
        plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.subplots_adjust(right=0.78)
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(benchmark_path, dpi=180, bbox_inches="tight")
        plt.close()


def _loads_trajectory(value) -> list:
    try:
        return json.loads(value)
    except Exception:
        return []


def _mean_curve(curves: list[list]) -> list[float]:
    y = []
    max_len = max(len(curve) for curve in curves)
    for pos in range(max_len):
        vals = [curve[pos][1] for curve in curves if pos < len(curve)]
        y.append(mean(vals))
    return y
