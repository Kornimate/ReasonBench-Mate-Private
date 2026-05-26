import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


HIGHLIGHT_METHOD_COLORS = {
    "heterogeneous_foa": "#d55e00",
    "heterogeneus_foa": "#d55e00",
    "reagents": "#0072b2",
}
DEFAULT_BAR_COLOR = "#8a8a8a"

METRIC_AXIS_LABELS = {
    "score_per_method_effort": "score per method effort",
    "normalized_action_entropy": "normalized entropy (0-1)",
    "calls_total": "calls (instances)",
    "total_tokens": "tokens",
    "total_cost": "cost (USD)",
    "score_per_dollar": "score per USD",
    "cost_per_solved": "cost per solved sample (USD)",
    "solved_per_dollar": "solved samples per USD",
    "solved_rate_per_dollar": "solved rate per USD",
    "cost_per_solved_rate_point": "cost per solved-rate point (USD)",
    "quality_mean": "quality score",
    "solved_rate": "solved rate (0-1)",
    "mean_solution_time": "mean solution time (seconds)",
    "mean_solved_solution_time": "mean solved solution time (seconds)",
    "clocktime_per_solved": "clocktime per solved sample (seconds)",
    "majority_agreement": "mean majority agreement (0-1)",
}


def ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with `pip install matplotlib`.") from exc
    return plt


def safe_filename(value: Any) -> str:
    text = "unknown" if value is None or pd.isna(value) else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "unknown"


def benchmark_output(output: Path, benchmark: Any) -> Path:
    if output.name == "overview.png":
        return output.with_name(f"{safe_filename(benchmark)}{output.suffix}")
    return output.with_name(f"{output.stem}_{safe_filename(benchmark)}{output.suffix}")


def metric_dir(output_dir: Path, metric: str) -> Path:
    return output_dir / safe_filename(metric)


def metric_plot(output_dir: Path, metric: str) -> Path:
    path = metric_dir(output_dir, metric)
    path.mkdir(parents=True, exist_ok=True)
    return path / "overview.png"


def metric_data(output_dir: Path, metric: str) -> Path:
    path = metric_dir(output_dir, metric)
    path.mkdir(parents=True, exist_ok=True)
    return path / "data.csv"


def subplot_grid_size(count: int) -> tuple[int, int]:
    columns = 4 if count > 3 else max(1, count)
    rows = math.ceil(count / columns)
    return rows, columns


def a4_landscape_size(rows: int) -> tuple[float, float]:
    return 11.69, max(8.27, rows * 2.6)


def bar_color(method: Any) -> str:
    return HIGHLIGHT_METHOD_COLORS.get(str(method), DEFAULT_BAR_COLOR)


def bar_colors(df: pd.DataFrame) -> list[str]:
    if "method" not in df.columns:
        return [DEFAULT_BAR_COLOR] * len(df)
    return [bar_color(method) for method in df["method"]]


def metric_axis_label(metric: str) -> str:
    return METRIC_AXIS_LABELS.get(metric, metric.replace("_", " "))


def add_method_highlight_legend(axis, df: pd.DataFrame) -> None:
    if "method" not in df.columns:
        return
    present = [method for method in ("heterogeneous_foa", "reagents") if method in set(df["method"].astype(str))]
    if not present:
        return
    from matplotlib.patches import Patch

    handles = [Patch(color=HIGHLIGHT_METHOD_COLORS[method], label=method) for method in present]
    axis.legend(handles=handles, fontsize=6, loc="best")


def draw_metric_bars(axis, labels: list[str], values: pd.Series, colors: list[str]) -> None:
    x_positions = list(range(len(labels)))
    axis.bar(x_positions, values, color=colors)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")


def benchmark_groups(df: pd.DataFrame):
    return list(df.groupby("benchmark", dropna=False))


def grouped_mean(df: pd.DataFrame, metric: str, group_by: list[str]) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    group_columns = [column for column in group_by if column in df.columns]
    if not group_columns:
        return pd.DataFrame()
    return df.dropna(subset=[metric]).groupby(group_columns, dropna=False)[metric].mean().reset_index()


def plot_bar_by_benchmark(df: pd.DataFrame, metric: str, title: str, output: Path) -> None:
    label_columns = [column for column in ("method",) if column in df.columns]
    plot_df = grouped_mean(df, metric, ["benchmark", *label_columns])
    if plot_df.empty:
        return
    plt = ensure_matplotlib()
    y_label = metric_axis_label(metric)
    groups = benchmark_groups(plot_df)
    rows, columns = subplot_grid_size(len(groups))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False)
    for axis, (benchmark, benchmark_df) in zip(axes.flat, groups):
        labels = [
            "\n".join(str(getattr(row, column)) for column in label_columns if hasattr(row, column))
            for row in benchmark_df.itertuples()
        ]
        draw_metric_bars(axis, labels, benchmark_df[metric], bar_colors(benchmark_df))
        axis.set_title(str(benchmark), fontsize=9)
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.set_ylabel(y_label, fontsize=8)
        add_method_highlight_legend(axis, benchmark_df)
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
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
        draw_metric_bars(plt.gca(), labels, benchmark_df[metric], bar_colors(benchmark_df))
        add_method_highlight_legend(plt.gca(), benchmark_df)
        plt.title(f"{title} - {benchmark}")
        plt.ylabel(y_label)
        plt.tight_layout()
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(benchmark_path, dpi=180)
        plt.close()
