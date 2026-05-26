import pandas as pd

from .log_metric_common import metric_plot, plot_bar_by_benchmark


def write(df: pd.DataFrame, output_dir) -> None:
    plot_bar_by_benchmark(df, "total_cost", "Total Cost", metric_plot(output_dir, "total_cost"))
