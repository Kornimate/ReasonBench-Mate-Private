import pandas as pd

from .log_metric_common import metric_plot, plot_bar_by_benchmark


def write(df: pd.DataFrame, output_dir) -> None:
    plot_bar_by_benchmark(
        df,
        "solved_rate_cost_ratio",
        "Solved-Rate/Cost Ratio",
        metric_plot(output_dir, "solved_rate_cost_ratio"),
    )
