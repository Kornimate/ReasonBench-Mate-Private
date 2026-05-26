#!/usr/bin/env python3
"""Pooled within-case clustering metrics for raw ReasonBench conversation logs.

The script extracts generated impression proposals with semantic_trajectory_metrics.py,
then pools every method's proposals for each case and clusters them into shared
semantic hypothesis families. Methods are compared on grounded cluster coverage,
discovery speed and redundancy under a sweep of cosine distance thresholds.

Diagnostic baseline (no model download):
  python cluster_trajectory_metrics.py mimic_rrs.zip --backend tfidf --output-dir results/clustering_tfidf

Clinical embedding run:
  python cluster_trajectory_metrics.py mimic_rrs.zip --backend sentence-transformers \
    --model FremyCompany/BioLORD-2023-M --output-dir results/clustering_biolord
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_trajectory_metrics", HERE / "semantic_trajectory_metrics.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(base)


def _as_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def cluster_case(vectors: np.ndarray, threshold: float) -> np.ndarray:
    if len(vectors) == 1:
        return np.zeros(1, dtype=int)
    # Embeddings are L2-normalized; Euclidean and cosine distance are monotonic:
    # ||a-b|| = sqrt(2 * cosine_distance(a,b)). This also tolerates zero-valued
    # latent baseline vectors, which are treated as uninformative points.
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=math.sqrt(2.0 * threshold),
        metric="euclidean",
        linkage="average",
    ).fit_predict(vectors)


def compute_threshold(
    proposals: pd.DataFrame,
    resources: pd.DataFrame,
    findings: list[str],
    proposal_vectors: np.ndarray,
    finding_vectors: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments: list[pd.DataFrame] = []
    cluster_rows: list[dict] = []
    metric_rows: list[dict] = []

    for case_id, group in proposals.groupby("case_id", sort=True):
        idx = group.index.to_numpy()
        X = _as_dense(proposal_vectors[idx])
        source = _as_dense(finding_vectors[int(case_id): int(case_id) + 1])
        labels = cluster_case(X, threshold)
        local = group.copy()
        local["threshold"] = threshold
        local["cluster_id"] = labels
        local["source_alignment"] = cosine_similarity(X, source).reshape(-1)
        assignments.append(local)

        case_clusters: dict[int, float] = {}
        for cluster_id, cgroup in local.groupby("cluster_id"):
            cidx_local = cgroup.index.to_numpy()
            cx = _as_dense(proposal_vectors[cidx_local])
            centroid = np.mean(cx, axis=0, keepdims=True)
            centroid_alignment = float(cosine_similarity(centroid, source)[0, 0])
            # Clip at zero: unrelated/opposing semantic modes should not increase coverage.
            weight = max(0.0, centroid_alignment)
            case_clusters[int(cluster_id)] = weight
            cluster_rows.append({
                "case_id": case_id,
                "threshold": threshold,
                "cluster_id": cluster_id,
                "cluster_size": len(cgroup),
                "method_count": cgroup["method"].nunique(),
                "centroid_alignment": centroid_alignment,
                "cluster_weight": weight,
                "methods": ";".join(sorted(cgroup["method"].unique())),
            })

        total_weight = sum(case_clusters.values()) or 1.0
        all_cluster_count = len(case_clusters)
        for method, mgroup in local.groupby("method", sort=False):
            covered = list(dict.fromkeys(mgroup.sort_values(["call_index", "response_index", "proposal_index"])["cluster_id"].tolist()))
            covered_weights = [case_clusters[int(label)] for label in covered]
            covered_weight = sum(covered_weights)
            counts = mgroup["cluster_id"].value_counts().to_numpy(dtype=float)
            probs = counts / counts.sum()
            effective_cluster_count = float(np.exp(-np.sum(probs * np.log(probs))))
            redundancy_rate = 1.0 - (len(covered) / len(mgroup))

            order = mgroup.sort_values(["call_index", "response_index", "proposal_index"])["cluster_id"].tolist()
            found: set[int] = set()
            discovery_curve = []
            for cluster_id in order:
                found.add(int(cluster_id))
                discovery_curve.append(sum(case_clusters[c] for c in found) / total_weight)
            discovery_auc = float(np.mean(discovery_curve))

            resource = resources[(resources.method == method) & (resources.case_id == case_id)].iloc[0]
            full_cost = math.log1p(float(resource.raw_completions))
            proposal_cost = math.log1p(len(mgroup))
            normalized_coverage = covered_weight / total_weight
            metric_rows.append({
                "method": method,
                "case_id": case_id,
                "threshold": threshold,
                "proposal_count": len(mgroup),
                "raw_completions": float(resource.raw_completions),
                "all_cluster_count": all_cluster_count,
                "covered_cluster_count": len(covered),
                "effective_cluster_count": effective_cluster_count,
                "redundancy_rate": redundancy_rate,
                "grounded_cluster_coverage": normalized_coverage,
                "grounded_cluster_discovery_auc": discovery_auc,
                "grounded_cluster_efficiency_full": normalized_coverage / full_cost,
                "grounded_cluster_efficiency_proposal": normalized_coverage / proposal_cost,
                "early_grounded_cluster_efficiency_full": discovery_auc / full_cost,
            })
    assign_df = pd.concat(assignments, ignore_index=True)
    clusters_df = pd.DataFrame(cluster_rows)
    cases_df = pd.DataFrame(metric_rows)
    return assign_df, clusters_df, cases_df


def aggregate(cases: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "proposal_count", "raw_completions", "all_cluster_count", "covered_cluster_count",
        "effective_cluster_count", "redundancy_rate", "grounded_cluster_coverage",
        "grounded_cluster_discovery_auc", "grounded_cluster_efficiency_full",
        "grounded_cluster_efficiency_proposal", "early_grounded_cluster_efficiency_full",
    ]
    by_threshold = cases.groupby(["threshold", "method"], as_index=False)[metric_cols].mean()
    ranking_metrics = [
        "grounded_cluster_coverage", "grounded_cluster_discovery_auc",
        "grounded_cluster_efficiency_full", "grounded_cluster_efficiency_proposal",
        "early_grounded_cluster_efficiency_full",
    ]
    for metric in ranking_metrics:
        by_threshold[f"rank_{metric}"] = by_threshold.groupby("threshold")[metric].rank(ascending=False, method="min")
    robustness_cols = ranking_metrics + ["redundancy_rate", "effective_cluster_count", "covered_cluster_count"] + [f"rank_{m}" for m in ranking_metrics]
    robustness = by_threshold.groupby("method", as_index=False)[robustness_cols].mean()
    robustness["mean_efficiency_rank"] = robustness[[
        "rank_grounded_cluster_efficiency_full",
        "rank_grounded_cluster_efficiency_proposal",
        "rank_early_grounded_cluster_efficiency_full",
    ]].mean(axis=1)
    return by_threshold, robustness.sort_values(["mean_efficiency_rank", "method"])


def render_summary(robustness: pd.DataFrame, by_threshold: pd.DataFrame, thresholds: list[float], backend: str, model: str | None) -> str:
    metric = "grounded_cluster_efficiency_full"
    label = "TF-IDF diagnostic baseline" if backend == "tfidf" else (model or "FremyCompany/BioLORD-2023-M")
    primary = robustness[["method", metric, f"rank_{metric}", "mean_efficiency_rank", "grounded_cluster_coverage", "redundancy_rate"]]
    return "\n".join([
        "# Pooled semantic-hypothesis clustering results", "",
        f"- Vector backend: `{label}`", f"- Cosine-distance thresholds swept: `{thresholds}`", "- TF-IDF runs apply bounded-vocabulary 32-dimensional truncated SVD (LSA) before clustering; clinical SentenceTransformer runs use normalized sentence embeddings directly.", "",
        "Each case is clustered jointly across all methods. A cluster represents a recurring semantic impression/hypothesis family. Cluster weight is its centroid similarity to source findings; methods earn coverage only once per cluster, so repeated paraphrases do not inflate the score.", "",
        "## Robust primary ranking", "",
        primary.to_markdown(index=False, floatfmt=".4f"), "",
        "`grounded_cluster_efficiency_full = grounded_cluster_coverage / ln(1 + all logged completions)`", "",
        "The robustness rank averages method ranks over all clustering thresholds. This reduces dependence on a single hand-selected clustering granularity.", "",
        "## Ranking at each threshold", "",
        by_threshold[["threshold", "method", metric, f"rank_{metric}"]].sort_values(["threshold", f"rank_{metric}"]).to_markdown(index=False, floatfmt=".4f"), "",
        "## Limitation", "",
        "A cluster can be aligned with the input findings without being clinically correct. Use this as a trajectory-efficiency measure and combine it with the independent benchmark correctness/quality score for final claims.", ""
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--backend", choices=["tfidf", "sentence-transformers"], default="tfidf")
    parser.add_argument("--model", default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.025, 0.05, 0.075, 0.10])
    parser.add_argument("--output-dir", type=Path, default=Path("cluster_metric_results"))
    parser.add_argument("--parsed-dir", type=Path, default=None, help="Reuse previous proposals.csv and metrics_by_case.csv to avoid reparsing long transcripts")
    args = parser.parse_args()

    if args.parsed_dir is not None:
        proposals = pd.read_csv(args.parsed_dir / "proposals.csv")
        resource_source = pd.read_csv(args.parsed_dir / "metrics_by_case.csv")
        resources = resource_source[["method", "case_id", "raw_calls", "raw_completions", "unassigned_call_overhead", "unassigned_completion_overhead"]].drop_duplicates()
        findings = proposals.drop_duplicates("case_id").sort_values("case_id")["findings"].tolist()
    else:
        logs = base.read_log_files(args.source)
        proposals, resources, findings = base.proposal_table(logs)
    texts = findings + proposals["proposal"].tolist()
    if args.backend == "tfidf":
        # Bounded-vocabulary latent semantic baseline: deterministic and quick to rerun.
        sparse = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=2, max_features=10000).fit_transform(texts)
        n_components = min(32, sparse.shape[0] - 1, sparse.shape[1] - 1)
        embedded = normalize(TruncatedSVD(n_components=n_components, n_iter=2, random_state=42).fit_transform(sparse))
    else:
        embedded = base.embed_texts(texts, args.backend, args.model)
    finding_vectors = embedded[: len(findings)]
    proposal_vectors = embedded[len(findings):]
    proposals = proposals.reset_index(drop=True)

    all_assign, all_clusters, all_cases = [], [], []
    for threshold in args.thresholds:
        assignments, clusters, cases = compute_threshold(
            proposals, resources, findings, proposal_vectors, finding_vectors, threshold
        )
        all_assign.append(assignments); all_clusters.append(clusters); all_cases.append(cases)
    assignments = pd.concat(all_assign, ignore_index=True)
    clusters = pd.concat(all_clusters, ignore_index=True)
    cases = pd.concat(all_cases, ignore_index=True)
    by_threshold, robustness = aggregate(cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments[["method", "case_id", "call_index", "response_index", "proposal_index", "threshold", "cluster_id", "source_alignment"]].to_csv(args.output_dir / "cluster_assignments.csv", index=False)
    clusters.to_csv(args.output_dir / "clusters_by_case.csv", index=False)
    cases.to_csv(args.output_dir / "cluster_metrics_by_case.csv", index=False)
    by_threshold.to_csv(args.output_dir / "cluster_metrics_by_method_threshold.csv", index=False)
    robustness.to_csv(args.output_dir / "cluster_metrics_robustness.csv", index=False)
    summary = render_summary(robustness, by_threshold, args.thresholds, args.backend, args.model)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(robustness[["method", "grounded_cluster_efficiency_full", "rank_grounded_cluster_efficiency_full", "mean_efficiency_rank", "grounded_cluster_coverage", "redundancy_rate"]].to_string(index=False))
    print(f"\nWrote clustering outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
