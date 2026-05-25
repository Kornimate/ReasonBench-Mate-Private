import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any
from omegaconf import OmegaConf

import pandas as pd


METHOD_TITLES = {
    "IO Method Information:",
    "CoT Method Information:",
    "CoT-SC Method Information:",
    "ReAct Method Information:",
    "FoA Method Information:",
    "Heterogeneous FoA Method Information:",
    "ToT-BFS Method Information:",
    "ToT-DFS Method Information:",
    "GoT Method Information:",
    "RAP Method Information:",
    "ReAgents Method Information:",
    "Runtime Agent Distribution Information:",
}

EVENT_PATTERN = re.compile(r"^\s*(?P<label>[A-Z0-9_]+)\s+(?P<payload>\{.*\})\s*$")
INFO_PREFIX_PATTERN = re.compile(r"^(?:INFO:[^:]+:)?(?P<message>.*)$")
SAMPLE_PATTERN = re.compile(r"Sample\s+(?P<sample>\d+):\s+solved=(?P<solved>True|False)\s+score=(?P<score>-?\d+(?:\.\d+)?)")
AVG_PATTERN = re.compile(r"Average correctness:\s*(?P<score>-?\d+(?:\.\d+)?)")
USAGE_PATTERN = re.compile(
    r"^(?P<kind>Calls|Tokens|Cost) "
    r"\((?P<scope>total|saved by cacher|saved by deduplicator)\):\s*(?P<value>.+)$"
)


@dataclass
class MethodLog:
    path: Path
    model: str | None = None
    benchmark: str | None = None
    method: str | None = None
    split: str | None = None
    repeat: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    avg_score: float | None = None
    calls_total: int | None = None
    calls_saved_by_cacher: int | None = None
    calls_saved_by_deduplicator: int | None = None
    input_tokens: float | None = None
    output_tokens: float | None = None
    cached_tokens: float | None = None
    input_tokens_saved_by_cacher: float | None = None
    output_tokens_saved_by_cacher: float | None = None
    input_tokens_saved_by_deduplicator: float | None = None
    output_tokens_saved_by_deduplicator: float | None = None
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None
    input_cost_saved_by_cacher: float | None = None
    output_cost_saved_by_cacher: float | None = None
    total_cost_saved_by_cacher: float | None = None
    input_cost_saved_by_deduplicator: float | None = None
    output_cost_saved_by_deduplicator: float | None = None
    total_cost_saved_by_deduplicator: float | None = None


def strip_log_prefix(line: str) -> str:
    match = INFO_PREFIX_PATTERN.match(line.rstrip("\n"))
    return match.group("message").strip() if match else line.strip()


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def set_usage_metric(parsed: MethodLog, kind: str, scope: str, value: str) -> None:
    suffix = "" if scope == "total" else f"_saved_by_{scope.removeprefix('saved by ').replace(' ', '_')}"

    if kind == "Calls":
        try:
            setattr(parsed, f"calls{suffix}" if suffix else "calls_total", int(value))
        except ValueError:
            return
        return

    try:
        usage = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return
    if not isinstance(usage, dict):
        return

    if kind == "Tokens":
        if suffix:
            setattr(parsed, f"input_tokens{suffix}", as_float(usage.get("in")))
            setattr(parsed, f"output_tokens{suffix}", as_float(usage.get("out")))
        else:
            parsed.input_tokens = as_float(usage.get("in"))
            parsed.output_tokens = as_float(usage.get("out"))
            parsed.cached_tokens = as_float(usage.get("cached"))
        return

    if kind == "Cost":
        if suffix:
            setattr(parsed, f"input_cost{suffix}", as_float(usage.get("in")))
            setattr(parsed, f"output_cost{suffix}", as_float(usage.get("out")))
            setattr(parsed, f"total_cost{suffix}", as_float(usage.get("total")))
        else:
            parsed.input_cost = as_float(usage.get("in"))
            parsed.output_cost = as_float(usage.get("out"))
            parsed.total_cost = as_float(usage.get("total"))


def infer_context(path: Path) -> dict[str, Any]:
    parts = path.parts
    context: dict[str, Any] = {
        "model": None,
        "benchmark": None,
        "method": None,
        "split": None,
        "repeat": None,
    }
    try:
        logs_idx = parts.index("logs")
    except ValueError:
        logs_idx = -1

    if logs_idx >= 0 and len(parts) >= logs_idx + 5:
        if len(parts) > logs_idx + 1 and parts[logs_idx + 1] == "raw_calls":
            context["model"] = parts[logs_idx + 3] if len(parts) > logs_idx + 3 else None
            context["benchmark"] = parts[logs_idx + 4] if len(parts) > logs_idx + 4 else None
        else:
            context["model"] = parts[logs_idx + 2]
            context["benchmark"] = parts[logs_idx + 3]

    stem = path.stem
    repeat_match = re.search(r"_(\d+)$", stem)
    if repeat_match:
        context["repeat"] = int(repeat_match.group(1))
        stem = stem[: repeat_match.start()]

    pieces = stem.split("_")
    known_splits = {"single", "mini", "train", "validation", "test"}
    if pieces and (pieces[-1] in known_splits or pieces[-1].startswith("n[")):
        context["split"] = pieces[-1]
        context["method"] = "_".join(pieces[:-1])
    elif len(pieces) >= 2:
        context["split"] = pieces[-1]
        context["method"] = "_".join(pieces[:-1])
    else:
        context["method"] = stem

    return context


def parse_method_log(path: Path) -> MethodLog:
    context = infer_context(path)
    parsed = MethodLog(path=path, **context)

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_log_prefix(raw_line)
        if not line:
            continue

        sample_match = SAMPLE_PATTERN.search(line)
        if sample_match:
            parsed.samples.append(
                {
                    "sample": int(sample_match.group("sample")),
                    "solved": sample_match.group("solved") == "True",
                    "score": float(sample_match.group("score")),
                }
            )
            continue

        avg_match = AVG_PATTERN.search(line)
        if avg_match:
            parsed.avg_score = float(avg_match.group("score"))
            continue

        usage_match = USAGE_PATTERN.match(line)
        if usage_match:
            set_usage_metric(
                parsed,
                usage_match.group("kind"),
                usage_match.group("scope"),
                usage_match.group("value"),
            )
            continue

        event_match = EVENT_PATTERN.match(line)
        if not event_match:
            continue

        try:
            payload = json.loads(event_match.group("payload"))
        except json.JSONDecodeError:
            continue

        payload["_label"] = event_match.group("label")
        parsed.events.append(payload)

    return parsed


def iter_method_logs(logs_dir: Path) -> list[MethodLog]:
    candidates = []
    for subdir in ("simple", "repeats", "cached"):
        root = logs_dir / subdir
        if root.exists():
            candidates.extend(root.rglob("*.log"))
    return [parse_method_log(path) for path in candidates]


def entropy(counts: Counter | dict[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def normalized_entropy(counts: Counter | dict[Any, int]) -> float:
    nonzero = sum(1 for count in counts.values() if count)
    if nonzero <= 1:
        return 0.0
    return entropy(counts) / math.log2(nonzero)


def _summary_values(summary: dict[str, Any] | None) -> list[float]:
    if not isinstance(summary, dict):
        return []
    values = summary.get("values")
    if isinstance(values, list):
        return [float(value) for value in values if isinstance(value, (int, float))]
    return []


def extract_step(event: dict[str, Any]) -> int | None:
    for key in ("step", "iteration", "depth"):
        value = event.get(key)
        if isinstance(value, int):
            return value
    return None


def final_score(log: MethodLog) -> float | None:
    if log.avg_score is not None:
        return log.avg_score
    if log.samples:
        return mean(sample["score"] for sample in log.samples)
    result_scores = []
    for event in log.events:
        terminal = event.get("terminal")
        if isinstance(terminal, dict):
            result_scores.extend(_summary_values(terminal.get("scores")))
    return max(result_scores) if result_scores else None


def event_effort(event: dict[str, Any]) -> float:
    effort = 0.0
    for key in ("proposal_count", "selected_count", "frontier_size", "width", "num_agents", "root_visits", "root_children"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            effort += max(0.0, float(value))
    for key in ("actions", "generated", "aggregated"):
        summary = event.get(key)
        if isinstance(summary, dict) and isinstance(summary.get("count"), (int, float)):
            effort += float(summary["count"])
    return effort


def effort_to_solution(logs: list[MethodLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        steps = [extract_step(event) for event in log.events if extract_step(event) is not None]
        solved_steps = [
            extract_step(event)
            for event in log.events
            if event.get("solved") is True and extract_step(event) is not None
        ]
        effort = sum(event_effort(event) for event in log.events)
        score = final_score(log)
        rows.append(
            {
                "model": log.model,
                "benchmark": log.benchmark,
                "method": log.method,
                "split": log.split,
                "repeat": log.repeat,
                "final_score": score,
                "event_count": len(log.events),
                "method_effort": effort,
                "steps_observed": max(steps) + 1 if steps else 0,
                "steps_to_first_solution": min(solved_steps) + 1 if solved_steps else None,
                "calls_total": log.calls_total,
                "score_per_method_effort": (score / effort if score is not None and effort > 0 else None),
                "score_per_call": (score / log.calls_total if score is not None and log.calls_total else None),
            }
        )
    return pd.DataFrame(rows)


def add_optional_values(*values: float | int | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None


def safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def usage_metrics(logs: list[MethodLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        score = final_score(log)
        total_tokens = add_optional_values(log.input_tokens, log.output_tokens)
        tokens_saved_by_cacher = add_optional_values(log.input_tokens_saved_by_cacher, log.output_tokens_saved_by_cacher)
        tokens_saved_by_deduplicator = add_optional_values(
            log.input_tokens_saved_by_deduplicator,
            log.output_tokens_saved_by_deduplicator,
        )
        total_tokens_with_saved = add_optional_values(total_tokens, tokens_saved_by_cacher, tokens_saved_by_deduplicator)
        total_cost_with_saved = add_optional_values(
            log.total_cost,
            log.total_cost_saved_by_cacher,
            log.total_cost_saved_by_deduplicator,
        )
        rows.append(
            {
                "model": log.model,
                "benchmark": log.benchmark,
                "method": log.method,
                "split": log.split,
                "repeat": log.repeat,
                "final_score": score,
                "calls_total": log.calls_total,
                "calls_saved_by_cacher": log.calls_saved_by_cacher,
                "calls_saved_by_deduplicator": log.calls_saved_by_deduplicator,
                "calls_without_cache_or_dedup": add_optional_values(
                    log.calls_total,
                    log.calls_saved_by_cacher,
                    log.calls_saved_by_deduplicator,
                ),
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "cached_tokens": log.cached_tokens,
                "total_tokens": total_tokens,
                "input_tokens_saved_by_cacher": log.input_tokens_saved_by_cacher,
                "output_tokens_saved_by_cacher": log.output_tokens_saved_by_cacher,
                "tokens_saved_by_cacher": tokens_saved_by_cacher,
                "input_tokens_saved_by_deduplicator": log.input_tokens_saved_by_deduplicator,
                "output_tokens_saved_by_deduplicator": log.output_tokens_saved_by_deduplicator,
                "tokens_saved_by_deduplicator": tokens_saved_by_deduplicator,
                "total_tokens_with_saved": total_tokens_with_saved,
                "token_savings_ratio": safe_ratio(
                    add_optional_values(tokens_saved_by_cacher, tokens_saved_by_deduplicator),
                    total_tokens_with_saved,
                ),
                "input_cost": log.input_cost,
                "output_cost": log.output_cost,
                "total_cost": log.total_cost,
                "input_cost_saved_by_cacher": log.input_cost_saved_by_cacher,
                "output_cost_saved_by_cacher": log.output_cost_saved_by_cacher,
                "total_cost_saved_by_cacher": log.total_cost_saved_by_cacher,
                "input_cost_saved_by_deduplicator": log.input_cost_saved_by_deduplicator,
                "output_cost_saved_by_deduplicator": log.output_cost_saved_by_deduplicator,
                "total_cost_saved_by_deduplicator": log.total_cost_saved_by_deduplicator,
                "total_cost_with_saved": total_cost_with_saved,
                "cost_savings_ratio": safe_ratio(
                    add_optional_values(log.total_cost_saved_by_cacher, log.total_cost_saved_by_deduplicator),
                    total_cost_with_saved,
                ),
                "score_per_call": safe_ratio(score, log.calls_total),
                "score_per_1k_tokens": safe_ratio(score, total_tokens / 1000 if total_tokens else None),
                "score_per_dollar": safe_ratio(score, log.total_cost),
                "cost_per_call": safe_ratio(log.total_cost, log.calls_total),
                "tokens_per_call": safe_ratio(total_tokens, log.calls_total),
            }
        )
    return pd.DataFrame(rows)


def exploration_diversity(logs: list[MethodLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        action_counts: Counter[str] = Counter()
        unique_ratios = []
        for event in log.events:
            for key in ("actions", "generated", "aggregated"):
                summary = event.get(key)
                if not isinstance(summary, dict):
                    continue
                total = summary.get("count")
                unique = summary.get("unique")
                if isinstance(total, (int, float)) and total:
                    unique_ratios.append(float(unique or 0) / float(total))
                for action, count in summary.get("most_common", []):
                    action_counts[str(action)] += int(count)
        rows.append(
            {
                "model": log.model,
                "benchmark": log.benchmark,
                "method": log.method,
                "split": log.split,
                "repeat": log.repeat,
                "action_entropy": entropy(action_counts),
                "normalized_action_entropy": normalized_entropy(action_counts),
                "mean_unique_action_ratio": mean(unique_ratios) if unique_ratios else None,
                "observed_action_mass": sum(action_counts.values()),
            }
        )
    return pd.DataFrame(rows)


def convergence_auc(logs: list[MethodLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        by_step: dict[int, list[float]] = defaultdict(list)
        solved_step = None
        for event in log.events:
            step = extract_step(event)
            if step is None:
                continue
            for key in ("values", "new_values", "old_values"):
                by_step[step].extend(_summary_values(event.get(key)))
            terminal = event.get("terminal")
            if isinstance(terminal, dict):
                by_step[step].extend(_summary_values(terminal.get("scores")))
            if event.get("solved") is True and solved_step is None:
                solved_step = step

        trajectory = []
        running_best = 0.0
        for step in sorted(by_step):
            if by_step[step]:
                running_best = max(running_best, max(by_step[step]))
            trajectory.append((step, running_best))

        if trajectory:
            denom = max(1, trajectory[-1][0] + 1)
            auc = sum(value for _, value in trajectory) / denom
            final_best = trajectory[-1][1]
        else:
            auc = None
            final_best = None

        rows.append(
            {
                "model": log.model,
                "benchmark": log.benchmark,
                "method": log.method,
                "split": log.split,
                "repeat": log.repeat,
                "convergence_auc": auc,
                "final_logged_best": final_best,
                "solved_step": solved_step,
                "trajectory": json.dumps(trajectory),
            }
        )
    return pd.DataFrame(rows)


RAW_USER_MARKER = "********\n* USER *\n********"
RAW_N_PATTERN = re.compile(r"\* N:\s*(?P<n>\d+)\s*\*")
RAW_RESPONSE_PATTERN = re.compile(r"\* RESPONSE\s+(?P<num>\d+)\s*\*")


def parse_raw_call_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split(RAW_USER_MARKER)
    calls = []
    for chunk in chunks[1:]:
        n_match = RAW_N_PATTERN.search(chunk)
        if not n_match:
            continue
        prompt = chunk[: n_match.start()].strip()
        response_area = chunk[n_match.end():]
        response_matches = list(RAW_RESPONSE_PATTERN.finditer(response_area))
        responses = []
        for i, match in enumerate(response_matches):
            start = match.end()
            end = response_matches[i + 1].start() if i + 1 < len(response_matches) else response_area.find("=" * 20, start)
            if end == -1:
                end = len(response_area)
            response = response_area[start:end].strip().strip("*").strip()
            if response:
                responses.append(response)
        if responses:
            calls.append({"path": path, "prompt": prompt, "n": int(n_match.group("n")), "responses": responses})
    return calls


def normalize_answer(response: str) -> str:
    cleaned = response.strip()
    match = re.search(r"\b([ABC])\b", cleaned.upper())
    if match:
        return match.group(1)
    cleaned = re.sub(r"\s+", " ", cleaned.lower())
    return cleaned[:160]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def mean_pairwise_edit_distance(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            denom = max(len(texts[i]), len(texts[j]), 1)
            distances.append(levenshtein(texts[i], texts[j]) / denom)
    return mean(distances) if distances else None


def raw_response_consistency(raw_dir: Path) -> pd.DataFrame:
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    prompt_examples = {}
    for path in raw_dir.rglob("*.log") if raw_dir.exists() else []:
        context = infer_context(path)
        for call in parse_raw_call_log(path):
            prompt_key = hashlib.sha256(call["prompt"].encode("utf-8")).hexdigest()[:16]
            group_key = (
                context.get("model"),
                context.get("benchmark"),
                context.get("method"),
                context.get("split"),
                context.get("repeat"),
                prompt_key,
            )
            prompt_examples[prompt_key] = call["prompt"][:240]
            grouped[group_key].extend(call["responses"])

    rows = []
    for group_key, responses in grouped.items():
        model, benchmark, method, split, repeat, prompt_key = group_key
        answer_counts = Counter(normalize_answer(response) for response in responses)
        majority = answer_counts.most_common(1)[0][1] if answer_counts else 0
        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "method": method,
                "split": split,
                "repeat": repeat,
                "prompt_key": prompt_key,
                "prompt_preview": prompt_examples.get(prompt_key, ""),
                "response_count": len(responses),
                "unique_answer_count": len(answer_counts),
                "answer_entropy": entropy(answer_counts),
                "normalized_answer_entropy": normalized_entropy(answer_counts),
                "majority_agreement": majority / len(responses) if responses else None,
                "mean_pairwise_edit_distance": mean_pairwise_edit_distance(responses),
                "mean_response_length": mean(len(response) for response in responses) if responses else None,
            }
        )
    return pd.DataFrame(rows)


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
    return output.with_name(f"{output.stem}_{safe_filename(benchmark)}{output.suffix}")


def grouped_mean(df: pd.DataFrame, metric: str, group_by: list[str]) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    return (
        df.dropna(subset=[metric])
        .groupby(group_by, dropna=False)[metric]
        .mean()
        .reset_index()
    )


def plot_bar_by_benchmark(df: pd.DataFrame, metric: str, title: str, output: Path) -> None:
    plot_df = grouped_mean(df, metric, ["benchmark", "method"])
    if plot_df.empty:
        return
    plt = ensure_matplotlib()
    for benchmark, benchmark_df in plot_df.groupby("benchmark", dropna=False):
        benchmark_df = benchmark_df.sort_values("method")
        labels = [str(method) for method in benchmark_df["method"]]
        plt.figure(figsize=(max(8, len(labels) * 0.55), 5))
        plt.bar(labels, benchmark_df[metric])
        plt.title(f"{title} - {benchmark}")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(benchmark_output(output, benchmark), dpi=180)
        plt.close()


def plot_convergence_by_benchmark(df: pd.DataFrame, output: Path) -> None:
    if df.empty or "trajectory" not in df.columns:
        return
    plt = ensure_matplotlib()
    for benchmark, benchmark_df in df.groupby("benchmark", dropna=False):
        plt.figure(figsize=(9, 5))
        has_curve = False
        for method, group in benchmark_df.groupby("method", dropna=False):
            curves = []
            for value in group["trajectory"]:
                try:
                    trajectory = json.loads(value)
                except Exception:
                    trajectory = []
                if trajectory:
                    curves.append(trajectory)
            if not curves:
                continue
            max_len = max(len(curve) for curve in curves)
            y = []
            for pos in range(max_len):
                vals = [curve[pos][1] for curve in curves if pos < len(curve)]
                y.append(mean(vals))
            plt.plot(range(len(y)), y, marker="o", label=str(method))
            has_curve = True
        if not has_curve:
            plt.close()
            continue
        plt.title(f"Logged Best-Score Convergence - {benchmark}")
        plt.xlabel("Logged step position")
        plt.ylabel("running best logged score")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(benchmark_output(output, benchmark), dpi=180)
        plt.close()


def plot_raw_consistency(df: pd.DataFrame, output: Path) -> None:
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
    plt.xlabel("majority agreement")
    plt.ylabel("mean pairwise edit distance")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_raw_agreement_by_method_and_benchmark(df: pd.DataFrame, output: Path) -> None:
    if df.empty or "method" not in df.columns:
        return
    plot_df = (
        df.dropna(subset=["majority_agreement"])
        .groupby(["benchmark", "method"], dropna=False)["majority_agreement"]
        .mean()
        .reset_index()
    )
    if plot_df.empty:
        return
    plt = ensure_matplotlib()
    for benchmark, benchmark_df in plot_df.groupby("benchmark", dropna=False):
        benchmark_df = benchmark_df.sort_values("method")
        labels = [str(method) for method in benchmark_df["method"]]
        plt.figure(figsize=(max(8, len(labels) * 0.55), 5))
        plt.bar(labels, benchmark_df["majority_agreement"])
        plt.title(f"Raw Call Majority Agreement - {benchmark}")
        plt.ylabel("mean majority agreement")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(benchmark_output(output, benchmark), dpi=180)
        plt.close()


def write_outputs(logs_dir: Path, raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_logs = iter_method_logs(logs_dir)

    effort = effort_to_solution(method_logs)
    diversity = exploration_diversity(method_logs)
    convergence = convergence_auc(method_logs)
    usage = usage_metrics(method_logs)
    raw = raw_response_consistency(raw_dir)

    effort.to_csv(output_dir / "effort_to_solution.csv", index=False)
    diversity.to_csv(output_dir / "exploration_diversity.csv", index=False)
    convergence.to_csv(output_dir / "convergence_auc.csv", index=False)
    usage.to_csv(output_dir / "usage_metrics.csv", index=False)
    raw.to_csv(output_dir / "raw_response_consistency.csv", index=False)

    plot_bar_by_benchmark(
        effort,
        "score_per_method_effort",
        "Score per Method Effort",
        output_dir / "effort_to_solution.png",
    )
    plot_bar_by_benchmark(
        diversity,
        "normalized_action_entropy",
        "Exploration Diversity",
        output_dir / "exploration_diversity.png",
    )
    plot_convergence_by_benchmark(convergence, output_dir / "convergence_auc.png")
    plot_bar_by_benchmark(usage, "calls_total", "Total Calls", output_dir / "calls_total.png")
    plot_bar_by_benchmark(usage, "total_tokens", "Total Tokens", output_dir / "total_tokens.png")
    plot_bar_by_benchmark(usage, "total_cost", "Total Cost", output_dir / "total_cost.png")
    plot_bar_by_benchmark(usage, "score_per_dollar", "Score per Dollar", output_dir / "score_per_dollar.png")
    plot_raw_consistency(raw, output_dir / "raw_response_consistency.png")
    plot_raw_agreement_by_method_and_benchmark(raw, output_dir / "raw_majority_agreement.png")


def main() -> None:
    config = OmegaConf.load("actions_config.yaml")
    write_outputs(
        logs_dir=Path(config.actions.visualize.log_path),
        raw_dir=Path(config.actions.visualize.raw_path),
        output_dir=Path(config.actions.visualize.output_path),
    )


if __name__ == "__main__":
    main()
