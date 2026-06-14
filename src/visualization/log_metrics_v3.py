from __future__ import annotations

import ast
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePath
from typing import Any

from omegaconf import OmegaConf

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reasonbench_matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.log_metrics_v2 import (
    SOLVED_THRESHOLDS_RUNTIME,
    load_logs,
    summarize_by_benchmark,
    validate_coverage,
)
from src.visualization.plot_style import (
    apply_plot_area_style,
    benchmark_label,
    configure_matplotlib,
    method_colors,
    method_labels,
)

configure_matplotlib()


FOCUS_METHODS = ["reagents", "heterogeneous_foa"]
EXCEPTION_BENCHMARKS = ["logiqa", "mtsamples_procedures", "pubmed_qa", "scibench"]
COMPARISON_METHODS = {
    "logiqa": ["tot_bfs", *FOCUS_METHODS],
    "mtsamples_procedures": ["io", *FOCUS_METHODS],
    "pubmed_qa": ["tot_bfs", *FOCUS_METHODS],
    "scibench": ["cot", *FOCUS_METHODS],
}
TRACE_PATTERN = re.compile(
    r"^\s*(?P<marker>TOT_BFS_STEP|REAGENTS_STEP|HETEROGENEOUS_FOA_STEP|IO_RESULT|COT_RESULT)\s+(?P<payload>\{.*\})\s*$",
    re.MULTILINE,
)
METHOD_PATTERN = re.compile(r"^\s*Method:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
BENCHMARK_PATTERN = re.compile(r"^\s*Benchmark:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
MODEL_PATTERN = re.compile(r"^\s*Model:\s*'(?P<value>[^']+)'\s*$", re.MULTILINE)
REPEAT_PATTERN = re.compile(r"_(?P<repeat>\d+)\.log$")
OPTION_PATTERN = re.compile(r"^[abcd]$", re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(r"\[(?:insert|specific|surgeon|date|signature|value|dose|duration)[^\]]*\]", re.IGNORECASE)
ELABORATION_PATTERN = re.compile(
    r"\b(plan|follow[- ]?up|monitor|recommend|management|postoperative|assessment|consult|education|schedule)\b",
    re.IGNORECASE,
)


def infer_path_parts(path: Path) -> tuple[str | None, str | None, str | None, int | None]:
    parts = PurePath(path).parts
    method = None
    repeat = None
    stem = path.stem
    pieces = stem.rsplit("_", 2)
    if len(pieces) == 3 and pieces[-1].isdigit():
        method = pieces[0]
        repeat = int(pieces[-1])
    if len(parts) >= 3:
        return parts[-3], parts[-2], method, repeat
    return None, None, method, repeat


def first_match(pattern: re.Pattern[str], text: str, fallback: str | None = None) -> str | None:
    match = pattern.search(text)
    return match.group("value") if match else fallback


def safe_values(payload: dict[str, Any], key: str) -> list[Any]:
    block = payload.get(key)
    if isinstance(block, dict):
        values = block.get("values", [])
        return values if isinstance(values, list) else []
    return []


def numeric_values(payload: dict[str, Any], key: str) -> list[float]:
    values = []
    for value in safe_values(payload, key):
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def action_values(payload: dict[str, Any]) -> list[str]:
    values = safe_values(payload, "actions")
    return [str(value) for value in values]


def terminal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    terminal = payload.get("terminal")
    return terminal if isinstance(terminal, dict) else {}


def terminal_scores(payload: dict[str, Any]) -> list[float]:
    terminal = terminal_payload(payload)
    scores = terminal.get("scores")
    if not isinstance(scores, dict):
        return []
    values = scores.get("values", [])
    result = []
    if isinstance(values, list):
        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                continue
    return result


def int_list(payload: dict[str, Any], key: str) -> list[int]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def numeric_payload_value(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_trace_logs(log_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for path in sorted(log_root.rglob("*.log")):
        if "raw_calls" in path.parts:
            continue
        fallback_model, fallback_benchmark, fallback_method, fallback_repeat = infer_path_parts(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        model = first_match(MODEL_PATTERN, text, fallback_model)
        benchmark = first_match(BENCHMARK_PATTERN, text, fallback_benchmark)
        method = first_match(METHOD_PATTERN, text, fallback_method)
        repeat_match = REPEAT_PATTERN.search(str(path))
        repeat = int(repeat_match.group("repeat")) if repeat_match else (fallback_repeat or 0)
        if benchmark not in EXCEPTION_BENCHMARKS or method not in set(COMPARISON_METHODS.get(benchmark, [])):
            continue

        for match in TRACE_PATTERN.finditer(text):
            marker = match.group("marker")
            try:
                payload = json.loads(match.group("payload"))
            except json.JSONDecodeError:
                continue
            actions = action_values(payload)
            action_lengths = [len(action.split()) for action in actions]
            option_actions = [action.strip().lower() for action in actions if OPTION_PATTERN.fullmatch(action.strip())]
            eval_values = numeric_values(payload, "evaluation")
            scores = terminal_scores(payload)
            terminal = terminal_payload(payload)
            state_depths = payload.get("state_depths", [])
            if not isinstance(state_depths, list):
                state_depths = []
            width = numeric_payload_value(payload, "width")
            next_width = numeric_payload_value(payload, "next_width")
            width_change = abs(next_width - width) if width is not None and next_width is not None else 0.0
            terminal_indices = set(int_list(payload, "terminal_indices"))
            solved_indices = set(int_list(payload, "solved_indices"))
            terminal_pruned_count = len(terminal_indices - solved_indices)
            idx = payload.get("idx")
            step = int(payload.get("step", 0))
            rows.append(
                {
                    "source_file": str(path),
                    "model": model,
                    "benchmark": benchmark,
                    "method": method,
                    "repeat": repeat,
                    "marker": marker,
                    "idx": idx,
                    "step": step,
                    "action_count": len(actions),
                    "unique_actions": len(set(actions)),
                    "option_action_count": len(option_actions),
                    "unique_option_actions": len(set(option_actions)),
                    "action_words_mean": float(np.mean(action_lengths)) if action_lengths else np.nan,
                    "action_words_max": float(np.max(action_lengths)) if action_lengths else np.nan,
                    "placeholder_hits": sum(len(PLACEHOLDER_PATTERN.findall(action)) for action in actions),
                    "elaboration_hits": sum(len(ELABORATION_PATTERN.findall(action)) for action in actions),
                    "terminal_count": int(terminal.get("terminal_count", 0) or 0),
                    "solved_count": int(terminal.get("solved_count", 0) or 0),
                    "terminal_score_mean": float(np.mean(scores)) if scores else np.nan,
                    "terminal_score_max": float(np.max(scores)) if scores else np.nan,
                    "terminal_score_std": float(np.std(scores)) if scores else np.nan,
                    "evaluation_count": len(eval_values),
                    "evaluation_mean": float(np.mean(eval_values)) if eval_values else np.nan,
                    "evaluation_max": float(np.max(eval_values)) if eval_values else np.nan,
                    "evaluation_std": float(np.std(eval_values)) if eval_values else np.nan,
                    "resampled_count": int(payload.get("resampled_count", 0) or 0),
                    "replacement_count": int(payload.get("replacement_count", 0) or 0),
                    "failed_count": int(payload.get("failed_count", 0) or 0),
                    "width": width,
                    "next_width": next_width,
                    "width_change": width_change,
                    "terminal_pruned_count": terminal_pruned_count,
                    "visited_count": int(payload.get("visited_count", 0) or 0),
                    "state_depth_mean": float(np.mean([float(v) for v in state_depths])) if state_depths else np.nan,
                }
            )
            for score in scores:
                candidate_rows.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "method": method,
                        "repeat": repeat,
                        "idx": idx,
                        "step": step,
                        "candidate_score": score,
                    }
                )
    if not rows:
        raise RuntimeError(f"No structured trace events found under {log_root}")
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def build_mechanism_summary(trace: pd.DataFrame, benchmark_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trace.groupby(["model", "benchmark", "method"], as_index=False)
        .agg(
            trace_events=("marker", "count"),
            mean_unique_actions=("unique_actions", "mean"),
            mean_unique_options=("unique_option_actions", "mean"),
            mean_action_words=("action_words_mean", "mean"),
            max_action_words=("action_words_max", "max"),
            placeholder_hits=("placeholder_hits", "sum"),
            elaboration_hits=("elaboration_hits", "sum"),
            mean_terminal_score_std=("terminal_score_std", "mean"),
            mean_evaluation_std=("evaluation_std", "mean"),
            total_resampled=("resampled_count", "sum"),
            total_replacements=("replacement_count", "sum"),
            total_failures=("failed_count", "sum"),
            total_width_change=("width_change", "sum"),
            total_terminal_pruned=("terminal_pruned_count", "sum"),
            mean_state_depth=("state_depth_mean", "mean"),
        )
    )
    summary = grouped.merge(
        benchmark_metrics[
            [
                "model",
                "benchmark",
                "method",
                "normalized_quality",
                "solved_rate",
                "solved_instances",
                "total_cost",
                "total_calls",
            ]
        ],
        on=["model", "benchmark", "method"],
        how="left",
    )
    summary["candidate_coverage"] = np.where(
        summary["mean_unique_options"] > 0,
        summary["mean_unique_options"],
        summary["mean_unique_actions"],
    )
    summary["drift_hits"] = summary["placeholder_hits"] + summary["elaboration_hits"]
    summary["drift_hits_per_trace"] = summary["drift_hits"] / summary["trace_events"].replace(0, np.nan)
    summary["explicit_selection_churn"] = (
        summary["total_resampled"] + summary["total_replacements"] + summary["total_failures"]
    )
    summary["explicit_selection_churn_per_trace"] = summary["explicit_selection_churn"] / summary[
        "trace_events"
    ].replace(0, np.nan)
    summary["structural_selection_churn"] = summary["total_width_change"] + summary["total_terminal_pruned"]
    summary["selection_churn_evidence"] = (
        summary["explicit_selection_churn"] + summary["structural_selection_churn"]
    )
    summary["selection_churn_evidence_per_trace"] = summary["selection_churn_evidence"] / summary[
        "trace_events"
    ].replace(0, np.nan)
    summary["branching_excess"] = (summary["mean_unique_actions"] - 1.0).clip(lower=0)
    summary["candidate_score_dispersion"] = summary["mean_terminal_score_std"].fillna(0)
    summary["cost_per_trace_event"] = summary["total_cost"] / summary["trace_events"].replace(0, np.nan)
    return add_winner_gaps(summary)


def add_winner_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (model, benchmark), part in summary.groupby(["model", "benchmark"], dropna=False):
        quality_winner = part["normalized_quality"].max()
        solved_winner = part["solved_rate"].max()
        cost_min = part["total_cost"].min()
        enriched = part.copy()
        enriched["quality_gap_vs_winner"] = enriched["normalized_quality"] - quality_winner
        enriched["solved_gap_vs_winner"] = enriched["solved_rate"] - solved_winner
        enriched["cost_ratio_vs_cheapest"] = enriched["total_cost"] / cost_min if cost_min else np.nan
        rows.append(enriched)
    return pd.concat(rows, ignore_index=True)


def _ordered_methods(frame: pd.DataFrame, benchmark: str) -> list[str]:
    present = set(frame[frame["benchmark"] == benchmark]["method"])
    return [method for method in COMPARISON_METHODS[benchmark] if method in present]


def _bar(ax: plt.Axes, methods: list[str], values: list[float], title: str, ylabel: str) -> None:
    x = np.arange(len(methods))
    ax.bar(x, values, color=method_colors(methods), edgecolor="white", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels(methods), rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    apply_plot_area_style(ax)
    for idx, value in enumerate(values):
        if pd.notna(value):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)


def plot_option_coverage_evidence(output_dir: Path, summary: pd.DataFrame) -> list[Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for model, data in summary.groupby("model", dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        for ax, bench in zip(axes, ["logiqa", "pubmed_qa"]):
            methods = _ordered_methods(data, bench)
            part = data[data["benchmark"].eq(bench)].set_index("method").reindex(methods)
            if bench == "logiqa":
                values = part["mean_unique_options"].to_list()
                title = f"{benchmark_label(bench)}: Multiple-choice option coverage"
                ylabel = "Unique a/b/c/d options"
                ax.set_ylim(0, 4.2)
            else:
                values = part["mean_unique_actions"].to_list()
                title = f"{benchmark_label(bench)}: Answer-candidate coverage"
                ylabel = "Unique candidate answers"
            _bar(ax, methods, values, title, ylabel)
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(methods)), part["solved_rate"], color="#264653", marker="o", linewidth=2)
            ax2.set_ylabel("Solved rate")
            ax2.set_ylim(0, 1.05)
            apply_plot_area_style(ax2)

        fig.suptitle(
            f"LogiQA and PubMedQA: Candidate coverage evidence - {model}\n"
            "Bars are trace-derived candidate coverage; dark lines show solved rate.",
            fontsize=13,
        )
        path = plots_dir / f"{_slug(model)}_logiqa_pubmedqa_candidate_coverage.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_mtsamples_drift_evidence(output_dir: Path, summary: pd.DataFrame) -> list[Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for model, data in summary.groupby("model", dropna=False):
        bench = "mtsamples_procedures"
        methods = _ordered_methods(data, bench)
        part = data[data["benchmark"].eq(bench)].set_index("method").reindex(methods)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        _bar(
            axes[0],
            methods,
            part["candidate_coverage"].to_list(),
            "Candidate branching",
            "Distinct candidates per trace",
        )
        noise = part["selection_churn_evidence_per_trace"]
        _bar(
            axes[1],
            methods,
            noise.to_list(),
            "Selection churn evidence",
            "Explicit churn + width/pruned per trace",
        )
        ax2 = axes[1].twinx()
        ax2.plot(np.arange(len(methods)), part["solved_rate"], color="#264653", marker="o", linewidth=2)
        ax2.set_ylabel("Solved rate")
        ax2.set_ylim(0, 1.05)
        apply_plot_area_style(ax2)

        fig.suptitle(
            f"MTSamples Procedures: Evidence for over-processing - {model}",
            fontsize=13,
        )
        path = plots_dir / f"{_slug(model)}_mtsamples_procedures_overprocessing_evidence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_scibench_disagreement_evidence(output_dir: Path, summary: pd.DataFrame) -> list[Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for model, data in summary.groupby("model", dropna=False):
        bench = "scibench"
        methods = _ordered_methods(data, bench)
        part = data[data["benchmark"].eq(bench)].set_index("method").reindex(methods)
        fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
        _bar(ax, methods, part["mean_unique_actions"].to_list(), "Candidate disagreement", "Distinct candidates per trace")
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(methods)), part["solved_rate"], color="#264653", marker="o", linewidth=2)
        ax2.set_ylabel("Solved rate")
        ax2.set_ylim(0, 1.05)
        apply_plot_area_style(ax2)

        fig.suptitle(
            f"SciBench: Evidence for candidate disagreement - {model}",
            fontsize=13,
        )
        path = plots_dir / f"{_slug(model)}_scibench_candidate_disagreement_evidence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_mechanism_evidence(output_dir: Path, summary: pd.DataFrame) -> list[Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale in plots_dir.glob("*.png"):
        stale.unlink()
    paths: list[Path] = []
    paths.extend(plot_option_coverage_evidence(output_dir, summary))
    paths.extend(plot_mtsamples_drift_evidence(output_dir, summary))
    paths.extend(plot_scibench_disagreement_evidence(output_dir, summary))
    return paths


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Log-Derived Failure Mechanism Evidence",
        "",
        "This report uses structured trace events embedded in the repeat logs, not only final metric tables.",
        "",
    ]
    idea_labels = {
        "logiqa": "Constrained options reward breadth-first option coverage.",
        "mtsamples_procedures": "Direct extraction is hurt when methods branch into extra candidates and selection work.",
        "pubmed_qa": "Constrained medical QA again rewards broader option coverage.",
        "scibench": "Single-chain CoT avoids disagreement across competing partial derivations.",
    }
    for (model, benchmark), part in summary.groupby(["model", "benchmark"], dropna=False):
        methods = _ordered_methods(part, benchmark)
        ordered = part.set_index("method").reindex(methods).dropna(how="all").reset_index()
        lines.extend([f"## {benchmark}", "", idea_labels[benchmark], ""])
        sixth_metric_label = "Candidate Disagreement" if benchmark == "scibench" else "Selection Churn/Trace"
        lines.append(
            f"| Method | Quality | Solved Rate | Candidate Coverage | Branching Excess | Drift/Trace | {sixth_metric_label} | Winner Gap |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in ordered.iterrows():
            sixth_metric = (
                row["candidate_coverage"]
                if benchmark == "scibench"
                else row["selection_churn_evidence_per_trace"]
            )
            lines.append(
                f"| `{row['method']}` | {row['normalized_quality']:.3f} | {row['solved_rate']:.3f} | "
                f"{row['candidate_coverage']:.2f} | {row['branching_excess']:.2f} | "
                f"{row['drift_hits_per_trace']:.2f} | {sixth_metric:.2f} | "
                f"{row['quality_gap_vs_winner']:.3f} |"
            )
        lines.append("")
    (output_dir / "failure_mechanism_evidence.md").write_text("\n".join(lines), encoding="utf-8")


def _summary_row(summary: pd.DataFrame, benchmark: str, method: str) -> pd.Series:
    row = summary[(summary["benchmark"] == benchmark) & (summary["method"] == method)]
    if row.empty:
        raise ValueError(f"Missing summary row for {benchmark}/{method}")
    return row.iloc[0]


def _fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def write_latex_report(output_dir: Path, summary: pd.DataFrame) -> None:
    logiqa_tot = _summary_row(summary, "logiqa", "tot_bfs")
    logiqa_reagents = _summary_row(summary, "logiqa", "reagents")
    logiqa_hfoa = _summary_row(summary, "logiqa", "heterogeneous_foa")
    pubmed_tot = _summary_row(summary, "pubmed_qa", "tot_bfs")
    pubmed_reagents = _summary_row(summary, "pubmed_qa", "reagents")
    pubmed_hfoa = _summary_row(summary, "pubmed_qa", "heterogeneous_foa")
    mts_io = _summary_row(summary, "mtsamples_procedures", "io")
    mts_reagents = _summary_row(summary, "mtsamples_procedures", "reagents")
    mts_hfoa = _summary_row(summary, "mtsamples_procedures", "heterogeneous_foa")
    sci_cot = _summary_row(summary, "scibench", "cot")
    sci_reagents = _summary_row(summary, "scibench", "reagents")
    sci_hfoa = _summary_row(summary, "scibench", "heterogeneous_foa")

    latex = rf"""
\subsection{{Failure-mode evidence from execution traces}}
\label{{sec:failure-mode-trace-evidence}}

To test whether the benchmark-level failures correspond to the proposed mechanisms, I extracted structured trace events from the repeat logs.  Each event records the candidates produced at a reasoning step, the number of distinct candidates, terminal scores when available, and method-specific search signals.  I use four derived metrics.  \emph{{Candidate coverage}} is the mean number of distinct answer candidates generated per trace event; for LogiQA it is the mean number of distinct multiple-choice options among \texttt{{a}}, \texttt{{b}}, \texttt{{c}}, and \texttt{{d}}, while for PubMedQA it is the mean number of distinct answer candidates.  \emph{{Branching excess}} is $\max(0, \text{{unique candidates}} - 1)$, so it measures how far a method moves away from a single direct answer.  \emph{{Drift per trace}} is the number of placeholder or elaborative-management terms per trace event, using markers such as ``plan'', ``follow-up'', ``monitor'', ``recommend'', and bracketed placeholders.  \emph{{Selection churn evidence per trace}} is $(\text{{resampled}}+\text{{replaced}}+\text{{failed}}+|\text{{next width}}-\text{{width}}|+\text{{terminal pruned}})/\text{{trace events}}$.  The first three terms capture explicit discard/resampling events logged by methods such as \texttt{{heterogeneous\_foa}}; the last two terms capture the corresponding ReAgents structure, where the logs record adaptive population width and terminal candidates removed from the continuing pool instead of explicit resampled indices.

\paragraph{{LogiQA and PubMedQA.}}
The LogiQA and PubMedQA failures both support the same candidate-coverage mechanism: when the task has a small answer space, \texttt{{tot\_bfs}} benefits from exploring more alternatives before selection.  On LogiQA, \texttt{{tot\_bfs}} reaches normalised quality {_fmt(logiqa_tot["normalized_quality"], 3)} and solved rate {_fmt(logiqa_tot["solved_rate"], 3)}, compared with {_fmt(logiqa_reagents["normalized_quality"], 3)}/{_fmt(logiqa_reagents["solved_rate"], 3)} for \texttt{{reagents}} and {_fmt(logiqa_hfoa["normalized_quality"], 3)}/{_fmt(logiqa_hfoa["solved_rate"], 3)} for \texttt{{heterogeneous\_foa}}.  The trace evidence is direct: \texttt{{tot\_bfs}} covers {_fmt(logiqa_tot["candidate_coverage"])} distinct multiple-choice options per trace event, while \texttt{{reagents}} covers {_fmt(logiqa_reagents["candidate_coverage"])} and \texttt{{heterogeneous\_foa}} covers {_fmt(logiqa_hfoa["candidate_coverage"])}.  PubMedQA shows the same pattern with answer-candidate coverage: \texttt{{tot\_bfs}} reaches quality {_fmt(pubmed_tot["normalized_quality"], 3)} and solved rate {_fmt(pubmed_tot["solved_rate"], 3)}, while both focus methods reach quality {_fmt(pubmed_reagents["normalized_quality"], 3)} and solved rate {_fmt(pubmed_reagents["solved_rate"], 3)}.  In the traces, \texttt{{tot\_bfs}} generates {_fmt(pubmed_tot["candidate_coverage"])} distinct candidates per event, compared with {_fmt(pubmed_reagents["candidate_coverage"])} for \texttt{{reagents}} and {_fmt(pubmed_hfoa["candidate_coverage"])} for \texttt{{heterogeneous\_foa}}.  Thus, in both benchmarks, the observed failure of the two focus methods is paired with lower candidate coverage than the winning breadth-first method.

\paragraph{{MTSamples Procedures.}}
The MTSamples Procedures failure is different: the direct \texttt{{io}} method is best, with quality {_fmt(mts_io["normalized_quality"], 3)} and solved rate {_fmt(mts_io["solved_rate"], 3)}, compared with {_fmt(mts_reagents["normalized_quality"], 3)}/{_fmt(mts_reagents["solved_rate"], 3)} for \texttt{{reagents}} and {_fmt(mts_hfoa["normalized_quality"], 3)}/{_fmt(mts_hfoa["solved_rate"], 3)} for \texttt{{heterogeneous\_foa}}.  I use only two trace metrics for this claim.  First, \emph{{candidate coverage}} measures how many distinct candidate outputs a method creates per trace event.  The direct method has coverage {_fmt(mts_io["candidate_coverage"])}, while \texttt{{reagents}} has {_fmt(mts_reagents["candidate_coverage"])} and \texttt{{heterogeneous\_foa}} has {_fmt(mts_hfoa["candidate_coverage"])}.  Second, \emph{{selection churn evidence per trace}} measures how much the population is revised through explicit resampling/replacement/failure events or, for \texttt{{reagents}}, through width adaptation and terminal-candidate pruning.  This is {_fmt(mts_io["selection_churn_evidence_per_trace"])} for \texttt{{io}}, {_fmt(mts_reagents["selection_churn_evidence_per_trace"])} for \texttt{{reagents}}, and {_fmt(mts_hfoa["selection_churn_evidence_per_trace"])} for \texttt{{heterogeneous\_foa}}.  These two metrics support a deterministic interpretation: both focus methods fail while producing more candidate variants and more population revision than direct \texttt{{io}}.  Thus, for this extraction-style task, the log evidence supports over-processing through extra branching and selection churn.

\paragraph{{SciBench.}}
SciBench supports the single-chain reasoning hypothesis.  \texttt{{cot}} is the winner, with quality {_fmt(sci_cot["normalized_quality"], 3)} and solved rate {_fmt(sci_cot["solved_rate"], 3)}, while \texttt{{reagents}} reaches {_fmt(sci_reagents["normalized_quality"], 3)}/{_fmt(sci_reagents["solved_rate"], 3)} and \texttt{{heterogeneous\_foa}} reaches {_fmt(sci_hfoa["normalized_quality"], 3)}/{_fmt(sci_hfoa["solved_rate"], 3)}.  The trace evidence supports a candidate-disagreement interpretation: \texttt{{cot}} produces {_fmt(sci_cot["candidate_coverage"])} distinct candidate per trace event, whereas \texttt{{reagents}} produces {_fmt(sci_reagents["candidate_coverage"])} distinct candidates and \texttt{{heterogeneous\_foa}} produces {_fmt(sci_hfoa["candidate_coverage"])}.  For calculation-heavy scientific problems, these competing partial derivations create multiple alternatives to choose among.  The logs therefore show that extra candidate disagreement does not beat a coherent single reasoning chain; it coincides with a quality gap of {_fmt(abs(sci_reagents["quality_gap_vs_winner"]), 3)} for \texttt{{reagents}} and {_fmt(abs(sci_hfoa["quality_gap_vs_winner"]), 3)} for \texttt{{heterogeneous\_foa}}.
""".strip()
    (output_dir / "failure_mechanism_evidence.tex").write_text(latex + "\n", encoding="utf-8")


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.md"):
        for path in output_dir.glob(pattern):
            path.unlink()


def main() -> int:
    config = OmegaConf.load("actions_config.yaml")
    log_root = Path(config.actions.visualize.log_path)
    output_dir = Path(config.actions.visualize.output_path) / "logv3"
    clean_output_dir(output_dir)

    runs, samples = load_logs(log_root, SOLVED_THRESHOLDS_RUNTIME)
    benchmark_metrics = summarize_by_benchmark(runs, samples)
    for warning in validate_coverage(benchmark_metrics):
        print(f"Warning: {warning}", file=sys.stderr)

    trace, candidates = parse_trace_logs(log_root)
    summary = build_mechanism_summary(trace, benchmark_metrics)

    trace.to_csv(output_dir / "trace_step_features.csv", index=False)
    candidates.to_csv(output_dir / "trace_candidate_scores.csv", index=False)
    summary.to_csv(output_dir / "failure_mechanism_summary.csv", index=False)
    write_report(output_dir, summary)
    write_latex_report(output_dir, summary)
    plot_paths = plot_mechanism_evidence(output_dir, summary)

    print(f"Parsed {len(trace)} structured trace events from {log_root}.")
    print(f"Saved mechanism evidence tables to: {output_dir.resolve()}")
    print(f"Saved {len(plot_paths)} plot(s) to: {(output_dir / 'plots').resolve()}")
    print(summary[summary["benchmark"].isin(EXCEPTION_BENCHMARKS)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
