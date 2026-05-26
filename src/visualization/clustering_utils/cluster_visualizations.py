#!/usr/bin/env python3
"""Visualization helpers for pooled cluster trajectory metrics.

This script does not rerun the clustering experiments. It reads the saved CSV
outputs and generates a small set of figures:

1) robust cluster-efficiency bar chart
2) cluster-efficiency vs threshold line chart
3) redundancy vs grounded coverage scatter chart
4) one illustrative semantic map for a representative case at a chosen threshold

The semantic map is intentionally only an *illustration*: it projects text into
2D after a dimensionality-reduction pipeline to mitigate the curse of high
 dimensionality. It should not be used as the quantitative basis of the metric.

Example:
  python cluster_visualizations.py \
    --results-dir results/clustering_tfidf \
    --proposals-csv ../semantic_metric_implementation/results/tfidf/proposals.csv \
    --output-dir results/clustering_tfidf/figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


FIG_DPI = 180


def load_tables(results_dir: Path, proposals_csv: Path):
    robustness = pd.read_csv(results_dir / "cluster_metrics_robustness.csv")
    by_threshold = pd.read_csv(results_dir / "cluster_metrics_by_method_threshold.csv")
    assignments = pd.read_csv(results_dir / "cluster_assignments.csv")
    cluster_cases = pd.read_csv(results_dir / "cluster_metrics_by_case.csv")
    clusters = pd.read_csv(results_dir / "clusters_by_case.csv")
    proposals = pd.read_csv(proposals_csv)
    semantic_dir = proposals_csv.parent
    semantic_cases = pd.read_csv(semantic_dir / "metrics_by_case.csv")
    semantic_methods = pd.read_csv(semantic_dir / "metrics_by_method.csv")
    return robustness, by_threshold, assignments, cluster_cases, clusters, proposals, semantic_cases, semantic_methods


def numeric_metric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [column for column in df.select_dtypes(include=["number"]).columns if column not in exclude]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def plot_method_bars(df: pd.DataFrame, metrics: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        plot_df = df[["method", metric]].dropna().sort_values(metric, ascending=True)
        if plot_df.empty:
            continue
        height = max(4.0, 0.45 * len(plot_df) + 1.5)
        plt.figure(figsize=(10, height))
        plt.barh(plot_df["method"].astype(str), plot_df[metric])
        plt.xlabel(metric)
        plt.ylabel("Method")
        plt.title(f"{prefix}: {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_filename(metric)}.png", dpi=FIG_DPI)
        plt.close()


def plot_threshold_lines(by_threshold: pd.DataFrame, metrics: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(by_threshold["method"].unique())
    for metric in metrics:
        if metric.startswith("rank_"):
            continue
        plt.figure(figsize=(10, 6))
        has_data = False
        for method in methods:
            sub = by_threshold[by_threshold["method"] == method].sort_values("threshold")
            sub = sub.dropna(subset=[metric])
            if sub.empty:
                continue
            has_data = True
            plt.plot(sub["threshold"], sub[metric], marker="o", label=method)
        if not has_data:
            plt.close()
            continue
        plt.xlabel("Cosine-distance threshold")
        plt.ylabel(metric)
        plt.title(f"{metric} across clustering thresholds")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_filename(metric)}.png", dpi=FIG_DPI)
        plt.close()


def plot_case_metric_distributions(df: pd.DataFrame, metrics: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(df["method"].unique())
    for metric in metrics:
        values = [df[df["method"] == method][metric].dropna().to_numpy() for method in methods]
        values = [value for value in values if len(value)]
        labels = [method for method in methods if len(df[df["method"] == method][metric].dropna())]
        if not values:
            continue
        width = max(10.0, 0.55 * len(labels))
        plt.figure(figsize=(width, 6))
        plt.boxplot(values, tick_labels=labels, vert=True, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{prefix}: case-level distribution of {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_filename(metric)}.png", dpi=FIG_DPI)
        plt.close()


def plot_cluster_table_distributions(clusters: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["cluster_size", "method_count", "centroid_alignment", "cluster_weight"]
    for metric in metrics:
        if metric not in clusters:
            continue
        plt.figure(figsize=(9, 5))
        plt.hist(clusters[metric].dropna(), bins=30)
        plt.xlabel(metric)
        plt.ylabel("Cluster count")
        plt.title(f"Distribution of {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_filename(metric)}.png", dpi=FIG_DPI)
        plt.close()


def plot_assignment_alignment(assignments: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for method in sorted(assignments["method"].unique()):
        sub = assignments[assignments["method"] == method]["source_alignment"].dropna()
        if sub.empty:
            continue
        plt.hist(sub, bins=30, alpha=0.35, label=method)
    plt.xlabel("Proposal source alignment")
    plt.ylabel("Proposal count")
    plt.title("Distribution of proposal-to-source alignment")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "source_alignment_by_method.png", dpi=FIG_DPI)
    plt.close()


def plot_robust_efficiency(robustness: pd.DataFrame, out_path: Path) -> None:
    df = robustness.sort_values("grounded_cluster_efficiency_full", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(df["method"], df["grounded_cluster_efficiency_full"])
    plt.xlabel("Grounded cluster efficiency (full)")
    plt.ylabel("Method")
    plt.title("Robust grounded cluster efficiency by method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def plot_efficiency_vs_threshold(by_threshold: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    methods = sorted(by_threshold["method"].unique())
    for method in methods:
        sub = by_threshold[by_threshold["method"] == method].sort_values("threshold")
        plt.plot(sub["threshold"], sub["grounded_cluster_efficiency_full"], marker="o", label=method)
    plt.xlabel("Cosine-distance threshold")
    plt.ylabel("Grounded cluster efficiency (full)")
    plt.title("Cluster efficiency across clustering granularities")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def plot_coverage_vs_redundancy(robustness: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    x = robustness["redundancy_rate"].to_numpy()
    y = robustness["grounded_cluster_coverage"].to_numpy()
    plt.scatter(x, y)
    for _, row in robustness.iterrows():
        plt.annotate(str(row["method"]), (row["redundancy_rate"], row["grounded_cluster_coverage"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Redundancy rate")
    plt.ylabel("Grounded cluster coverage")
    plt.title("Coverage–redundancy trade-off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def choose_representative_case(assignments: pd.DataFrame, threshold: float) -> int:
    sub = assignments[np.isclose(assignments["threshold"], threshold)]
    counts = sub.groupby("case_id").size().sort_values(ascending=False)
    return int(counts.index[0])


def semantic_map_dataframe(assignments: pd.DataFrame, proposals: pd.DataFrame, threshold: float, case_id: int | None = None) -> tuple[pd.DataFrame, int]:
    if case_id is None:
        case_id = choose_representative_case(assignments, threshold)

    sub_assign = assignments[(np.isclose(assignments["threshold"], threshold)) & (assignments["case_id"] == case_id)].copy()
    key_cols = ["method", "case_id", "call_index", "response_index", "proposal_index"]
    sub = sub_assign.merge(
        proposals[key_cols + ["proposal", "findings"]],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    sub = sub.dropna(subset=["proposal"]).reset_index(drop=True)

    texts = sub["proposal"].astype(str).tolist()
    # Two-stage reduction to mitigate high-dimensional noise:
    # TF-IDF -> latent semantic subspace -> 2D PCA for plotting only.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, max_features=5000)
    X = tfidf.fit_transform(texts)
    n_samples, n_features = X.shape
    if n_samples < 2:
        coords = np.zeros((n_samples, 2), dtype=float)
    else:
        latent_dims = max(2, min(20, n_samples - 1, max(2, n_features - 1)))
        Z = TruncatedSVD(n_components=latent_dims, random_state=42).fit_transform(X)
        Z = normalize(Z)
        if Z.shape[1] > 2 and Z.shape[0] >= 2:
            coords = PCA(n_components=2, random_state=42).fit_transform(Z)
        elif Z.shape[1] == 2:
            coords = Z
        else:
            # Fallback for degenerate tiny cases
            coords = np.column_stack([Z[:, 0], np.zeros(Z.shape[0])]) if Z.shape[1] == 1 else np.zeros((Z.shape[0], 2))
    sub["x"] = coords[:, 0]
    sub["y"] = coords[:, 1]
    return sub, case_id


def plot_semantic_map(assignments: pd.DataFrame, proposals: pd.DataFrame, out_path: Path, threshold: float = 0.05, case_id: int | None = None) -> int:
    df, actual_case_id = semantic_map_dataframe(assignments, proposals, threshold, case_id)
    plt.figure(figsize=(10, 7))
    methods = sorted(df["method"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods)}

    for method in methods:
        sub = df[df["method"] == method]
        plt.scatter(sub["x"], sub["y"], marker=marker_map[method], s=50, label=method)

    # annotate cluster centers
    centers = df.groupby("cluster_id")[["x", "y"]].mean().reset_index()
    for _, row in centers.iterrows():
        plt.annotate(f"C{int(row['cluster_id'])}", (row["x"], row["y"]), fontsize=9)

    plt.xlabel("2D semantic component 1")
    plt.ylabel("2D semantic component 2")
    plt.title(f"Illustrative semantic cluster map for case {actual_case_id} at threshold {threshold}")
    plt.legend(loc="best", fontsize=8, ncol=2)
    note = (
        "Projection only: TF-IDF → latent semantic reduction → PCA(2D). "
        "Distances are illustrative and not the metric itself."
    )
    plt.figtext(0.5, 0.01, note, ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    return actual_case_id


def write_manifest(out_dir: Path, representative_case: int, threshold: float) -> None:
    text = "\n".join([
        "# Visualization outputs",
        "",
        "These figures are derived from the already saved clustering outputs.",
        "No clustering experiment was rerun.",
        "",
        "Files:",
        "- robust_efficiency_bar.png: robust full-efficiency ranking by method",
        "- efficiency_vs_threshold.png: sensitivity of full efficiency to clustering threshold",
        "- coverage_vs_redundancy.png: trade-off plot between coverage and repetition",
        f"- semantic_map_case_{representative_case}.png: illustrative 2D semantic map for case {representative_case} at threshold {threshold}",
        "- semantic_method_metrics/: one bar chart per semantic method-level metric",
        "- semantic_case_metric_distributions/: one boxplot per semantic case-level metric",
        "- cluster_robustness_metrics/: one bar chart per threshold-robust method-level cluster metric",
        "- cluster_threshold_metrics/: one line chart per threshold-sensitive cluster metric",
        "- cluster_case_metric_distributions/: one boxplot per cluster case-level metric",
        "- cluster_table_distributions/: histograms for cluster size, method sharing, and cluster grounding",
        "- assignment_distributions/: proposal assignment diagnostics, including source-alignment histograms",
        "",
        "High-dimensionality note:",
        "The semantic map uses a two-stage dimensionality reduction pipeline (TF-IDF → latent semantic projection → PCA to 2D) to reduce sparsity/noise before plotting. The 2D plot is for interpretation only and should not be used as the quantitative score.",
    ])
    (out_dir / "README_visualizations.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--proposals-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-threshold", type=float, default=0.05)
    parser.add_argument("--case-id", type=int, default=None)
    args = parser.parse_args()

    robustness, by_threshold, assignments, cluster_cases, clusters, proposals, semantic_cases, semantic_methods = load_tables(
        args.results_dir, args.proposals_csv
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_robust_efficiency(robustness, args.output_dir / "robust_efficiency_bar.png")
    plot_efficiency_vs_threshold(by_threshold, args.output_dir / "efficiency_vs_threshold.png")
    plot_coverage_vs_redundancy(robustness, args.output_dir / "coverage_vs_redundancy.png")

    plot_method_bars(
        semantic_methods,
        numeric_metric_columns(semantic_methods, {"case_id"}),
        args.output_dir / "semantic_method_metrics",
        "Semantic method metrics",
    )
    plot_case_metric_distributions(
        semantic_cases,
        numeric_metric_columns(semantic_cases, {"case_id"}),
        args.output_dir / "semantic_case_metric_distributions",
        "Semantic trajectory metrics",
    )
    plot_method_bars(
        robustness,
        numeric_metric_columns(robustness, set()),
        args.output_dir / "cluster_robustness_metrics",
        "Cluster robustness metrics",
    )
    plot_threshold_lines(
        by_threshold,
        numeric_metric_columns(by_threshold, {"threshold"}),
        args.output_dir / "cluster_threshold_metrics",
    )
    plot_case_metric_distributions(
        cluster_cases,
        numeric_metric_columns(cluster_cases, {"case_id", "threshold"}),
        args.output_dir / "cluster_case_metric_distributions",
        "Cluster trajectory metrics",
    )
    plot_cluster_table_distributions(clusters, args.output_dir / "cluster_table_distributions")
    plot_assignment_alignment(assignments, args.output_dir / "assignment_distributions")

    case_id = plot_semantic_map(assignments, proposals, args.output_dir / f"semantic_map_case_{args.case_id if args.case_id is not None else 'auto'}.png", threshold=args.semantic_threshold, case_id=args.case_id)

    # Rename auto file to actual case id for clarity.
    auto_path = args.output_dir / "semantic_map_case_auto.png"
    actual_path = args.output_dir / f"semantic_map_case_{case_id}.png"
    if auto_path.exists() and not actual_path.exists():
        auto_path.rename(actual_path)
    write_manifest(args.output_dir, case_id, args.semantic_threshold)
    print(f"Wrote visualizations to {args.output_dir}")
    print(f"Representative semantic-map case: {case_id}")


if __name__ == "__main__":
    main()
