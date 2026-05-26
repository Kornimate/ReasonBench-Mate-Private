from pathlib import Path

import pandas as pd

from .log_metric_common import ensure_matplotlib, metric_plot


def write(df: pd.DataFrame, output_dir: Path) -> None:
    output = metric_plot(output_dir, "raw_response_consistency")
    if df.empty:
        return
    plt = ensure_matplotlib()
    plt.figure(figsize=(8, 5))
    sizes = [max(20, count * 8) for count in df["response_count"]]
    plt.scatter(
        df["majority_agreement"],
        df["mean_pairwise_edit_distance"],
        s=sizes,
        c=df["normalized_answer_entropy"],
        cmap="viridis",
        alpha=0.75,
    )
    plt.colorbar(label="normalized answer entropy")
    plt.title("Raw Call Response Consistency")
    plt.xlabel("majority agreement (0-1)")
    plt.ylabel("mean pairwise edit distance (0-1)")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
