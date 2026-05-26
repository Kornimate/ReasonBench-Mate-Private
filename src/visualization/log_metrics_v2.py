from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator
from omegaconf import OmegaConf

# Use an OS-appropriate writable Matplotlib cache directory in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reasonbench_matplotlib_cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Runtime solved thresholds from the benchmark environments.
# src/tasks/mimic_rrs/environment.py: FINAL_SCORE_THRESHOLD = 4.0
# src/tasks/mtsamples_procedures/environment.py: FINAL_SCORE_THRESHOLD = 3.8
SOLVED_THRESHOLDS_RUNTIME: dict[str, float] = {
    "game24": 1.0,
    "hle": 1.0,
    "hotpotqa": 1.0,
    "humaneval": 1.0,
    "logiqa": 1.0,
    "matharena": 1.0,
    "mimic_rrs": 4.0,
    "mtsamples_procedures": 3.8,
    "pubmed_qa": 1.0,
    "scibench": 1.0,
    "sonnetwriting": 1.0,
}

# Retained only to reproduce the current visualization/log_metrics.py behavior.
SOLVED_THRESHOLDS_LEGACY: dict[str, float] = {
    **SOLVED_THRESHOLDS_RUNTIME,
    "mimic_rrs": 3.5,
    "mtsamples_procedures": 3.5,
}

# Scores for medical summarization are grader scores on a five-point scale.
# Other listed benchmarks have a quality scale in [0, 1].
QUALITY_MAX_SCORE: dict[str, float] = {
    "mimic_rrs": 5.0,
    "mtsamples_procedures": 5.0,
}

QUALITY_PATTERN = re.compile(r"Quality:\s*.*?Correct:\s*(?P<values>\[.*?\])", re.DOTALL)
METHOD_PATTERN = re.compile(r"^\s*Method:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
BENCHMARK_PATTERN = re.compile(r"^\s*Benchmark:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
MODEL_PATTERN = re.compile(r"^\s*Model:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
SPLIT_PATTERN = re.compile(r"^\s*Split:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
ALL_TABS_PATTERN = re.compile(r"All Tabs:\s*(?P<block>.*?)(?:\nDuration:|\Z)", re.DOTALL)
CALLS_PATTERN = re.compile(r"Calls \(total\):\s*(?P<value>\d+)")
COST_PATTERN = re.compile(r"Cost \(total\):\s*(?P<value>\{.*?\})")
TOKENS_PATTERN = re.compile(r"Tokens \(total\):\s*(?P<value>\{.*?\})")
REPEAT_PATTERN = re.compile(r"_(?P<repeat>\d+)\.log$")


@dataclass(frozen=True)
class LogSource:
    name: str
    text: str


def _literal_list(text: str) -> list[float]:
    values = ast.literal_eval(text)
    if not isinstance(values, list):
        raise ValueError("Expected quality values to be a list")
    return [float(value) for value in values]


def _literal_dict(text: str) -> dict[str, float]:
    values = ast.literal_eval(text)
    if not isinstance(values, dict):
        raise ValueError("Expected usage values to be a dictionary")
    return {str(key): float(value) for key, value in values.items()}


def _match_value(pattern: re.Pattern[str], text: str, fallback: str | None = None) -> str | None:
    match = pattern.search(text)
    return match.group("value") if match else fallback


def iter_log_sources(input_path: Path) -> Iterator[LogSource]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".log") and "/raw_calls/" not in name:
                    yield LogSource(name=name, text=archive.read(name).decode("utf-8", errors="ignore"))
        return

    if input_path.is_dir():
        for path in sorted(input_path.rglob("*.log")):
            if "raw_calls" not in path.parts:
                yield LogSource(name=str(path), text=path.read_text(encoding="utf-8", errors="ignore"))
        return

    raise FileNotFoundError(f"Input must be a ZIP file or directory: {input_path}")


def infer_from_path(name: str) -> tuple[str | None, str | None]:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    # Expected shape: .../repeats/<model>/<benchmark>/<method>_<split>_<repeat>.log
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return None, None


def parse_log(source: LogSource, thresholds: dict[str, float]) -> tuple[dict[str, object], list[dict[str, object]]]:
    fallback_model, fallback_benchmark = infer_from_path(source.name)
    method = _match_value(METHOD_PATTERN, source.text)
    benchmark = _match_value(BENCHMARK_PATTERN, source.text, fallback_benchmark)
    model = _match_value(MODEL_PATTERN, source.text, fallback_model)
    split = _match_value(SPLIT_PATTERN, source.text)
    repeat_match = REPEAT_PATTERN.search(source.name)
    repeat = int(repeat_match.group("repeat")) if repeat_match else 0

    if not method or not benchmark:
        raise ValueError(f"Could not identify method/benchmark from {source.name}")

    quality_match = QUALITY_PATTERN.search(source.text)
    if not quality_match:
        raise ValueError(f"No 'Quality: Correct' score array found in {source.name}")
    scores = _literal_list(quality_match.group("values"))
    if not scores:
        raise ValueError(f"Empty score array in {source.name}")

    usage_block_match = ALL_TABS_PATTERN.search(source.text)
    usage_text = usage_block_match.group("block") if usage_block_match else source.text
    calls_matches = list(CALLS_PATTERN.finditer(usage_text))
    cost_matches = list(COST_PATTERN.finditer(usage_text))
    token_matches = list(TOKENS_PATTERN.finditer(usage_text))
    calls_total = int(calls_matches[-1].group("value")) if calls_matches else np.nan
    costs = _literal_dict(cost_matches[-1].group("value")) if cost_matches else {}
    tokens = _literal_dict(token_matches[-1].group("value")) if token_matches else {}

    threshold = thresholds.get(benchmark, 1.0)
    maximum = QUALITY_MAX_SCORE.get(benchmark, 1.0)
    normalized = [score / maximum for score in scores]
    solved = [score >= threshold for score in scores]

    run_row: dict[str, object] = {
        "source_file": source.name,
        "model": model,
        "benchmark": benchmark,
        "method": method,
        "split": split,
        "repeat": repeat,
        "attempted_instances": len(scores),
        "solved_threshold": threshold,
        "quality_max_score": maximum,
        "quality_mean_raw": float(np.mean(scores)),
        "quality_mean_normalized": float(np.mean(normalized)),
        "solved_instances": int(sum(solved)),
        "solved_rate": float(np.mean(solved)),
        "total_cost": costs.get("total", np.nan),
        "total_calls": calls_total,
        "input_tokens": tokens.get("in", np.nan),
        "output_tokens": tokens.get("out", np.nan),
    }
    sample_rows = [
        {
            "model": model,
            "benchmark": benchmark,
            "method": method,
            "repeat": repeat,
            "sample_index": index,
            "score_raw": score,
            "score_normalized": normalized[index],
            "solved": solved[index],
        }
        for index, score in enumerate(scores)
    ]
    return run_row, sample_rows


def load_logs(input_path: Path, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    errors: list[str] = []
    for source in iter_log_sources(input_path):
        try:
            run, sample_rows = parse_log(source, thresholds)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        runs.append(run)
        samples.extend(sample_rows)
    if not runs:
        details = "\n".join(errors[:5])
        raise RuntimeError(f"No analysable log files found.\n{details}")
    if errors:
        print(f"Warning: skipped {len(errors)} unparseable logs.", file=sys.stderr)
        for error in errors[:5]:
            print(f"  - {error}", file=sys.stderr)
    return pd.DataFrame(runs), pd.DataFrame(samples)


def summarize_by_benchmark(runs: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    quality = (
        samples.groupby(["model", "benchmark", "method"], as_index=False)
        .agg(
            attempted_instances=("solved", "size"),
            solved_instances=("solved", "sum"),
            normalized_quality=("score_normalized", "mean"),
            raw_quality=("score_raw", "mean"),
        )
    )
    quality["solved_rate"] = quality["solved_instances"] / quality["attempted_instances"]
    usage = (
        runs.groupby(["model", "benchmark", "method"], as_index=False)
        .agg(log_files=("source_file", "count"), total_cost=("total_cost", "sum"), total_calls=("total_calls", "sum"))
    )
    return quality.merge(usage, on=["model", "benchmark", "method"], how="left")


def harmonic_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    denominator = a + b
    return np.where(denominator > 0, (2 * a * b) / denominator, 0.0)


def pareto_frontier(frame: pd.DataFrame) -> pd.Series:
    """Four-dimensional frontier: quality and solved up; cost and calls down."""
    result = []
    for idx, candidate in frame.iterrows():
        dominated = False
        for other_idx, other in frame.iterrows():
            if idx == other_idx:
                continue
            no_worse = (
                other["macro_normalized_quality"] >= candidate["macro_normalized_quality"]
                and other["solved_instances"] >= candidate["solved_instances"]
                and other["total_cost"] <= candidate["total_cost"]
                and other["total_calls"] <= candidate["total_calls"]
            )
            strictly_better = (
                other["macro_normalized_quality"] > candidate["macro_normalized_quality"]
                or other["solved_instances"] > candidate["solved_instances"]
                or other["total_cost"] < candidate["total_cost"]
                or other["total_calls"] < candidate["total_calls"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        result.append(not dominated)
    return pd.Series(result, index=frame.index)


def summarize_by_method(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model, part in benchmark.groupby("model", dropna=False):
        summary = (
            part.groupby("method", as_index=False)
            .agg(
                benchmarks=("benchmark", "nunique"),
                attempted_instances=("attempted_instances", "sum"),
                solved_instances=("solved_instances", "sum"),
                macro_normalized_quality=("normalized_quality", "mean"),
                macro_solved_rate=("solved_rate", "mean"),
                total_cost=("total_cost", "sum"),
                total_calls=("total_calls", "sum"),
            )
        )
        # Pooled rates weight benchmarks by their actual number of evaluated samples.
        pooled = (
            part.assign(quality_sum=lambda d: d["normalized_quality"] * d["attempted_instances"])
            .groupby("method", as_index=False)
            .agg(quality_sum=("quality_sum", "sum"))
        )
        summary = summary.merge(pooled, on="method")
        summary.insert(0, "model", model)
        summary["micro_normalized_quality"] = summary["quality_sum"] / summary["attempted_instances"]
        summary["overall_solved_rate"] = summary["solved_instances"] / summary["attempted_instances"]
        summary["harmonic_effectiveness"] = harmonic_mean(
            summary["macro_normalized_quality"], summary["macro_solved_rate"]
        )
        summary["cost_per_solved"] = summary["total_cost"] / summary["solved_instances"].replace(0, np.nan)
        summary["calls_per_solved"] = summary["total_calls"] / summary["solved_instances"].replace(0, np.nan)
        summary["quality_per_dollar"] = summary["macro_normalized_quality"] / summary["total_cost"].replace(0, np.nan)
        summary["quality_per_1k_calls"] = summary["macro_normalized_quality"] / (
            summary["total_calls"].replace(0, np.nan) / 1000
        )
        summary["solved_per_dollar"] = summary["solved_instances"] / summary["total_cost"].replace(0, np.nan)
        summary["solved_per_1k_calls"] = summary["solved_instances"] / (
            summary["total_calls"].replace(0, np.nan) / 1000
        )
        summary["rank_quality"] = summary["macro_normalized_quality"].rank(ascending=False, method="min")
        summary["rank_solved"] = summary["solved_instances"].rank(ascending=False, method="min")
        summary["rank_cost"] = summary["total_cost"].rank(ascending=True, method="min")
        summary["rank_calls"] = summary["total_calls"].rank(ascending=True, method="min")
        summary["performance_priority_composite_rank"] = (
            0.40 * summary["rank_quality"]
            + 0.30 * summary["rank_solved"]
            + 0.15 * summary["rank_cost"]
            + 0.15 * summary["rank_calls"]
        )
        summary["pareto_optimal"] = pareto_frontier(summary)
        summary = summary.drop(columns=["quality_sum"])
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["model", "macro_normalized_quality"], ascending=[True, False]
    )


def metric_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("macro_normalized_quality", False),
        ("micro_normalized_quality", False),
        ("solved_instances", False),
        ("overall_solved_rate", False),
        ("macro_solved_rate", False),
        ("harmonic_effectiveness", False),
        ("total_cost", True),
        ("total_calls", True),
        ("cost_per_solved", True),
        ("calls_per_solved", True),
        ("quality_per_dollar", False),
        ("quality_per_1k_calls", False),
        ("solved_per_dollar", False),
        ("solved_per_1k_calls", False),
        ("performance_priority_composite_rank", True),
    ]
    rows: list[pd.DataFrame] = []
    for model, model_frame in summary.groupby("model", dropna=False):
        for metric, ascending in specs:
            ranked = model_frame[["method", metric]].sort_values(metric, ascending=ascending).reset_index(drop=True)
            ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
            ranked.insert(0, "metric", metric)
            ranked.insert(0, "model", model)
            ranked = ranked.rename(columns={metric: "value"})
            rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def bootstrap_top_quality_difference(
    samples: pd.DataFrame, summary: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    """Paired, benchmark-stratified bootstrap for top-two macro quality methods."""
    if iterations <= 0:
        return pd.DataFrame()
    outputs: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for model, method_summary in summary.groupby("model", dropna=False):
        top_two = method_summary.nlargest(2, "macro_normalized_quality")["method"].tolist()
        if len(top_two) != 2:
            continue
        first, second = top_two
        model_samples = samples[samples["model"] == model]
        differences: list[np.ndarray] = []
        observed_by_benchmark: list[float] = []
        for benchmark, bench in model_samples.groupby("benchmark"):
            a = bench[bench["method"] == first].set_index(["repeat", "sample_index"])["score_normalized"]
            b = bench[bench["method"] == second].set_index(["repeat", "sample_index"])["score_normalized"]
            paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if paired.empty:
                continue
            delta = (paired["a"] - paired["b"]).to_numpy()
            observed_by_benchmark.append(float(delta.mean()))
            sample_idx = rng.integers(0, len(delta), size=(iterations, len(delta)))
            differences.append(delta[sample_idx].mean(axis=1))
        if not differences:
            continue
        boot_delta = np.vstack(differences).mean(axis=0)
        observed = float(np.mean(observed_by_benchmark))
        outputs.append(
            {
                "model": model,
                "higher_quality_method": first,
                "comparison_method": second,
                "observed_macro_quality_difference": observed,
                "ci_95_low": float(np.quantile(boot_delta, 0.025)),
                "ci_95_high": float(np.quantile(boot_delta, 0.975)),
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(outputs)


def validate_coverage(benchmark: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for model, model_part in benchmark.groupby("model", dropna=False):
        methods = model_part["method"].unique()
        for bench, bench_part in model_part.groupby("benchmark"):
            counts = bench_part.set_index("method")["attempted_instances"]
            if len(counts) != len(methods):
                warnings.append(f"{model}/{bench}: missing method logs ({len(counts)}/{len(methods)} present).")
            if counts.nunique() > 1:
                warnings.append(f"{model}/{bench}: methods have unequal attempted counts: {counts.to_dict()}")
    return warnings


def save_outputs(
    output_dir: Path,
    runs: pd.DataFrame,
    samples: pd.DataFrame,
    benchmark: pd.DataFrame,
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "run_level_metrics.csv", index=False)
    samples.to_csv(output_dir / "sample_level_scores.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_method_metrics.csv", index=False)
    summary.to_csv(output_dir / "method_summary_metrics.csv", index=False)
    rankings.to_csv(output_dir / "metric_rankings.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(output_dir / "top_quality_bootstrap.csv", index=False)



def _slug(value: object) -> str:
    """Create a safe filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def _format_axis(ax: plt.Axes, metric: str) -> None:
    """Format metric-specific axes without applying hard-coded visual styling."""
    if metric in {"macro_normalized_quality", "micro_normalized_quality", "overall_solved_rate",
                  "macro_solved_rate", "harmonic_effectiveness", "normalized_quality", "solved_rate"}:
        ax.set_ylim(0, 1.05)
    if "cost" in metric or "dollar" in metric:
        ax.set_ylabel("USD")
    elif "calls" in metric:
        ax.set_ylabel("Calls")
    elif "rate" in metric or "quality" in metric or metric == "harmonic_effectiveness":
        ax.set_ylabel("Score")
    else:
        ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.25)


def focus_benchmark_difficulty(benchmark: pd.DataFrame, focus_methods: list[str]) -> pd.DataFrame:
    """Summarize and order benchmark difficulty for selected comparison methods."""
    selected = benchmark[benchmark["method"].isin(focus_methods)].copy()
    missing = sorted(set(focus_methods) - set(selected["method"].unique()))
    if missing:
        raise ValueError(f"Focus method(s) absent from parsed logs: {missing}")
    expected = len(focus_methods)
    available = selected.groupby(["model", "benchmark"])["method"].nunique()
    incomplete = available[available != expected]
    if not incomplete.empty:
        raise ValueError(f"Not every focus method is available for all benchmarks: {incomplete.to_dict()}")

    rows: list[pd.DataFrame] = []
    for model, part in selected.groupby("model", dropna=False):
        result = (
            part.groupby("benchmark", as_index=False)
            .agg(
                instances_per_method=("attempted_instances", "first"),
                mean_focus_normalized_quality=("normalized_quality", "mean"),
                mean_focus_solved_rate=("solved_rate", "mean"),
                combined_focus_cost=("total_cost", "sum"),
                combined_focus_calls=("total_calls", "sum"),
            )
            .sort_values(["mean_focus_normalized_quality", "mean_focus_solved_rate", "benchmark"])
            .reset_index(drop=True)
        )
        result.insert(0, "model", model)
        result.insert(1, "difficulty_rank_by_quality", np.arange(1, len(result) + 1))
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def overall_metric_specs() -> list[tuple[str, str, bool]]:
    """Metrics plotted for all-method aggregate and per-benchmark views."""
    return [
        ("macro_normalized_quality", "Macro Normalized Quality", False),
        ("micro_normalized_quality", "Micro Normalized Quality", False),
        ("solved_instances", "Total Solved Instances", False),
        ("overall_solved_rate", "Overall (Micro) Solve Rate", False),
        ("macro_solved_rate", "Macro Solve Rate", False),
        ("harmonic_effectiveness", "Harmonic Effectiveness", False),
        ("total_cost", "Total Cost", True),
        ("total_calls", "Total Calls", True),
        ("cost_per_solved", "Cost per Solved Instance", True),
        ("calls_per_solved", "Calls per Solved Instance", True),
        ("quality_per_dollar", "Macro Quality per Dollar", False),
        ("quality_per_1k_calls", "Macro Quality per 1,000 Calls", False),
        ("solved_per_dollar", "Solved Instances per Dollar", False),
        ("solved_per_1k_calls", "Solved Instances per 1,000 Calls", False),
        ("performance_priority_composite_rank", "Performance-Priority Composite Rank", True),
    ]


def plot_overall_metrics(summary: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Write one aggregate bar chart for every method-level metric."""
    metrics = overall_metric_specs()
    paths: list[Path] = []
    overall_dir = plots_dir / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    for model, data in summary.groupby("model", dropna=False):
        model_slug = _slug(model)
        for metric, title, ascending in metrics:
            ordered = data.sort_values(metric, ascending=ascending)
            fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
            ax.bar(ordered["method"], ordered[metric])
            ax.set_title(f"{title} — {model}")
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=40)
            _format_axis(ax, metric)
            for idx, value in enumerate(ordered[metric]):
                if pd.notna(value):
                    label = f"{value:,.4f}" if isinstance(value, (float, np.floating)) else f"{value:,}"
                    ax.text(idx, value, label, ha="center", va="bottom", fontsize=8)
            path = overall_dir / f"{model_slug}_{metric}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_pareto_views(summary: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Plot quality against both resource dimensions and label Pareto-efficient methods."""
    pareto_dir = plots_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model, data in summary.groupby("model", dropna=False):
        for resource, xlabel in [("total_cost", "Total Cost (USD)"), ("total_calls", "Total Calls")]:
            fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
            ax.scatter(data[resource], data["macro_normalized_quality"], s=65)
            for _, row in data.iterrows():
                suffix = " *" if bool(row["pareto_optimal"]) else ""
                ax.annotate(
                    f"{row['method']}{suffix}",
                    (row[resource], row["macro_normalized_quality"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_title(f"Quality–Resource Trade-off — {model}\n* = Pareto-optimal")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Macro Normalized Quality")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            path = pareto_dir / f"{_slug(model)}_quality_vs_{resource}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_focus_per_benchmark(
    benchmark: pd.DataFrame, focus_methods: list[str], difficulty: pd.DataFrame, plots_dir: Path
) -> list[Path]:
    """Create one four-panel comparison plot per benchmark for focus methods."""
    paths: list[Path] = []
    output = plots_dir / "per_benchmark_focus_methods"
    output.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("normalized_quality", "Normalized Quality"),
        ("solved_rate", "Solved Rate"),
        ("total_cost", "Total Cost"),
        ("total_calls", "Total Calls"),
    ]
    selected = benchmark[benchmark["method"].isin(focus_methods)].copy()
    for model, model_diff in difficulty.groupby("model", dropna=False):
        ordered_benchmarks = model_diff.sort_values("difficulty_rank_by_quality")["benchmark"].tolist()
        part = selected[selected["model"] == model]
        for rank, bench_name in enumerate(ordered_benchmarks, start=1):
            values = part[part["benchmark"] == bench_name].set_index("method").reindex(focus_methods).reset_index()
            attempts = int(values["attempted_instances"].iloc[0])
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
            for ax, (metric, title) in zip(axes.flat, metrics):
                ax.bar(values["method"], values[metric])
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=22)
                _format_axis(ax, metric)
                for i, value in enumerate(values[metric]):
                    label = f"{value:.4f}" if metric not in {"total_calls"} else f"{int(value):,}"
                    ax.text(i, value, label, ha="center", va="bottom", fontsize=8)
            fig.suptitle(
                f"{bench_name} — {model}\nFocus-method difficulty rank {rank}/{len(ordered_benchmarks)}; "
                f"{attempts} evaluated instances per method",
                fontsize=12,
            )
            path = output / f"{rank:02d}_{_slug(model)}_{_slug(bench_name)}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_focus_dashboard(
    benchmark: pd.DataFrame, focus_methods: list[str], difficulty: pd.DataFrame, plots_dir: Path
) -> list[Path]:
    """Plot all benchmarks for selected methods in one four-panel dashboard."""
    dashboard_dir = plots_dir / "dashboards"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metrics = [
        ("normalized_quality", "Normalized Quality"),
        ("solved_rate", "Solved Rate"),
        ("total_cost", "Total Cost (USD)"),
        ("total_calls", "Total Calls"),
    ]
    selected = benchmark[benchmark["method"].isin(focus_methods)].copy()
    for model, model_diff in difficulty.groupby("model", dropna=False):
        order = model_diff.sort_values("difficulty_rank_by_quality")["benchmark"].tolist()
        part = selected[selected["model"] == model]
        x = np.arange(len(order))
        width = min(0.8 / len(focus_methods), 0.35)
        fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
        for ax, (metric, title) in zip(axes.flat, metrics):
            for index, method in enumerate(focus_methods):
                vals = part[part["method"] == method].set_index("benchmark").reindex(order)[metric]
                offset = (index - (len(focus_methods) - 1) / 2) * width
                ax.bar(x + offset, vals, width=width, label=method)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=45, ha="right")
            _format_axis(ax, metric)
            ax.legend()
        fig.suptitle(
            f"Focus Methods Across Benchmarks — {model}\n"
            "Benchmarks ordered hardest → easiest by mean focus-method normalized quality",
            fontsize=14,
        )
        path = dashboard_dir / f"{_slug(model)}_focus_methods_across_benchmarks.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_all_method_benchmark_dashboards(benchmark: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Create one large dashboard per metric, with one subplot per benchmark and all methods shown."""
    dashboard_dir = plots_dir / "dashboards"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metrics = [
        ("normalized_quality", "Normalized Quality"),
        ("solved_rate", "Solved Rate"),
        ("total_cost", "Total Cost"),
        ("total_calls", "Total Calls"),
    ]
    for model, part in benchmark.groupby("model", dropna=False):
        benchmarks = sorted(part["benchmark"].unique())
        methods = sorted(part["method"].unique())
        ncols = 3
        nrows = int(np.ceil(len(benchmarks) / ncols))
        for metric, title in metrics:
            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), constrained_layout=True)
            axes_array = np.array(axes).reshape(-1)
            for ax, bench_name in zip(axes_array, benchmarks):
                values = (
                    part[part["benchmark"] == bench_name]
                    .set_index("method")
                    .reindex(methods)
                    .reset_index()
                )
                ax.bar(values["method"], values[metric])
                ax.set_title(f"{bench_name} (n={int(values['attempted_instances'].iloc[0])})")
                ax.tick_params(axis="x", rotation=55, labelsize=7)
                _format_axis(ax, metric)
            for ax in axes_array[len(benchmarks):]:
                ax.axis("off")
            fig.suptitle(f"{title} by Benchmark and Method — {model}", fontsize=15)
            path = dashboard_dir / f"{_slug(model)}_all_benchmarks_subplots_{metric}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_all_method_benchmark_metric_plots(benchmark: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Create one standalone plot per benchmark/overall-style metric, with all methods shown."""
    output_dir = plots_dir / "per_benchmark_all_methods"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metrics = overall_metric_specs()
    for model, model_part in benchmark.groupby("model", dropna=False):
        methods = sorted(model_part["method"].unique())
        for bench_name, bench_part in model_part.groupby("benchmark", dropna=False):
            values = bench_part.set_index("method").reindex(methods).reset_index()
            values["macro_normalized_quality"] = values["normalized_quality"]
            values["micro_normalized_quality"] = values["normalized_quality"]
            values["overall_solved_rate"] = values["solved_rate"]
            values["macro_solved_rate"] = values["solved_rate"]
            values["harmonic_effectiveness"] = harmonic_mean(
                values["macro_normalized_quality"], values["macro_solved_rate"]
            )
            values["cost_per_solved"] = values["total_cost"] / values["solved_instances"].replace(0, np.nan)
            values["calls_per_solved"] = values["total_calls"] / values["solved_instances"].replace(0, np.nan)
            values["quality_per_dollar"] = values["macro_normalized_quality"] / values["total_cost"].replace(0, np.nan)
            values["quality_per_1k_calls"] = values["macro_normalized_quality"] / (
                values["total_calls"].replace(0, np.nan) / 1000
            )
            values["solved_per_dollar"] = values["solved_instances"] / values["total_cost"].replace(0, np.nan)
            values["solved_per_1k_calls"] = values["solved_instances"] / (
                values["total_calls"].replace(0, np.nan) / 1000
            )
            values["rank_quality"] = values["macro_normalized_quality"].rank(ascending=False, method="min")
            values["rank_solved"] = values["solved_instances"].rank(ascending=False, method="min")
            values["rank_cost"] = values["total_cost"].rank(ascending=True, method="min")
            values["rank_calls"] = values["total_calls"].rank(ascending=True, method="min")
            values["performance_priority_composite_rank"] = (
                0.40 * values["rank_quality"]
                + 0.30 * values["rank_solved"]
                + 0.15 * values["rank_cost"]
                + 0.15 * values["rank_calls"]
            )
            attempts = values["attempted_instances"].dropna()
            attempts_label = f"n={int(attempts.iloc[0])}" if not attempts.empty else "n=unknown"
            for metric, title, ascending in metrics:
                ordered = values.sort_values(metric, ascending=ascending, na_position="last")
                fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
                ax.bar(ordered["method"], ordered[metric])
                ax.set_title(f"{title} by Method - {bench_name} - {model} ({attempts_label})")
                ax.set_xlabel("Method")
                ax.tick_params(axis="x", rotation=45)
                _format_axis(ax, metric)
                for idx, value in enumerate(ordered[metric]):
                    if pd.notna(value):
                        label = f"{int(value):,}" if metric in {"solved_instances", "total_calls"} else f"{value:,.4f}"
                        ax.text(idx, value, label, ha="center", va="bottom", fontsize=8)
                path = output_dir / f"{_slug(model)}_{_slug(bench_name)}_{metric}.png"
                fig.savefig(path, dpi=180)
                plt.close(fig)
                paths.append(path)
    return paths


def generate_plots(
    output_dir: Path,
    benchmark: pd.DataFrame,
    summary: pd.DataFrame,
    focus_methods: list[str],
) -> tuple[list[Path], pd.DataFrame]:
    """Generate all requested plots and the focus-method difficulty table."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    difficulty = focus_benchmark_difficulty(benchmark, focus_methods)
    difficulty.to_csv(output_dir / "focus_methods_benchmark_difficulty.csv", index=False)
    paths: list[Path] = []
    paths.extend(plot_overall_metrics(summary, plots_dir))
    paths.extend(plot_pareto_views(summary, plots_dir))
    paths.extend(plot_focus_per_benchmark(benchmark, focus_methods, difficulty, plots_dir))
    paths.extend(plot_focus_dashboard(benchmark, focus_methods, difficulty, plots_dir))
    paths.extend(plot_all_method_benchmark_dashboards(benchmark, plots_dir))
    paths.extend(plot_all_method_benchmark_metric_plots(benchmark, plots_dir))
    return paths, difficulty

def print_results(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    columns = [
        "method",
        "attempted_instances",
        "solved_instances",
        "overall_solved_rate",
        "macro_normalized_quality",
        "total_cost",
        "total_calls",
        "performance_priority_composite_rank",
        "pareto_optimal",
    ]
    for model, data in summary.groupby("model", dropna=False):
        print(f"\nMODEL: {model}")
        formatted = data[columns].sort_values("macro_normalized_quality", ascending=False).copy()
        for col in ["overall_solved_rate", "macro_normalized_quality", "total_cost", "performance_priority_composite_rank"]:
            formatted[col] = formatted[col].map(lambda x: f"{x:.6f}")
        print(formatted.to_string(index=False))
    if not bootstrap.empty:
        print("\nTOP-TWO PAIRED BOOTSTRAP (macro normalized quality difference)")
        print(bootstrap.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to repeats.zip or an extracted directory containing .log files")
    parser.add_argument("--output-dir", type=Path, default=Path("metrics_output"), help="Directory for generated CSV tables")
    parser.add_argument(
        "--threshold-mode",
        choices=("runtime", "legacy"),
        default="runtime",
        help="Use benchmark runtime medical thresholds (recommended) or reproduce legacy visualization thresholds",
    )
    parser.add_argument("--bootstrap", type=int, default=50000, help="Bootstrap samples for top-two quality difference; 0 disables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    parser.add_argument("--print-table", action="store_true", help="Print the main summary table to stdout")
    parser.add_argument("--no-plots", action="store_true", help="Export CSV tables only; skip all PNG plot generation")
    parser.add_argument(
        "--focus-methods",
        nargs="+",
        default=["heterogeneous_foa", "reagents"],
        help="Methods used in dedicated per-benchmark comparison plots (default: heterogeneous_foa reagents)",
    )
    return parser


def main() -> int:
    config = OmegaConf.load("actions_config.yaml")

    # thresholds = SOLVED_THRESHOLDS_RUNTIME if args.threshold_mode == "runtime" else SOLVED_THRESHOLDS_LEGACY
    thresholds = SOLVED_THRESHOLDS_RUNTIME
    runs, samples = load_logs(Path(config.actions.visualize.log_path), thresholds)
    benchmark = summarize_by_benchmark(runs, samples)
    for warning in validate_coverage(benchmark):
        print(f"Warning: {warning}", file=sys.stderr)
    summary = summarize_by_method(benchmark)
    rankings = metric_rankings(summary)
    bootstrap = bootstrap_top_quality_difference(samples, summary, 10000, 42)
    save_outputs(Path(config.actions.visualize.output_path), runs, samples, benchmark, summary, rankings, bootstrap)

    print(f"Parsed {len(runs)} log files and {len(samples)} recorded sample scores.")
    print(f"Saved metric tables to: {Path(config.actions.visualize.output_path).resolve()}")
    
    plot_paths: list[Path] = []
    plot_paths, _ = generate_plots(Path(config.actions.visualize.output_path).resolve(), benchmark, summary, ["heterogeneous_foa", "reagents"]) # focus methods

    print(f"Parsed {len(runs)} log files and {len(samples)} recorded sample scores.")
    print(f"Saved metric tables to: {Path(config.actions.visualize.output_path).resolve()}")
    print(f"Saved {len(plot_paths)} PNG plots to: {(Path(config.actions.visualize.output_path) / 'plots').resolve()}")

    print_results(summary, bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
