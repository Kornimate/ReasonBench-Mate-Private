import pandas as pd

from .log_metric_common import metric_plot, plot_bar_by_benchmark


def write(df: pd.DataFrame, output_dir) -> None:
    plot_bar_by_benchmark(df, "clocktime_per_solved", "Clocktime per Solved", metric_plot(output_dir, "clocktime_per_solved"))
