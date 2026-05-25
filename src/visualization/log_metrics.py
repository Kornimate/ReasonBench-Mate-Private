import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
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
TOTAL_CLOCKTIME_PATTERN = re.compile(r"Duration:.*?Total clocktime \(in seconds\):\s*(?P<value>-?\d+(?:\.\d+)?)", re.DOTALL)
DURATIONS_PATTERN = re.compile(
    r"Duration:.*?Individual durations of each sample \(in seconds\):\s*(?P<values>\[.*?\])",
    re.DOTALL,
)
QUALITY_CORRECT_PATTERN = re.compile(r"Quality:.*?Correct:\s*(?P<values>\[.*?\])", re.DOTALL)
USAGE_PATTERN = re.compile(
    r"^(?P<kind>Calls|Tokens|Cost) "
    r"\((?P<scope>total|saved by cacher|saved by deduplicator)\):\s*(?P<value>.+)$"
)

SOLVED_THRESHOLDS = {
    "game24": 1.0,
    "hle": 1.0,
    "hotpotqa": 1.0,
    "humaneval": 1.0,
    "logiqa": 1.0,
    "matharena": 1.0,
    "mimic_rrs": 3.5,
    "mtsamples_procedures": 3.5,
    "pubmed_qa": 1.0,
    "scibench": 1.0,
    "sonnetwriting": 1.0,
}

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
    "quality_mean": "quality score",
    "solved_rate": "solved rate (0-1)",
    "mean_solution_time": "mean solution time (seconds)",
    "mean_solved_solution_time": "mean solved solution time (seconds)",
    "clocktime_per_solved": "clocktime per solved sample (seconds)",
    "majority_agreement": "mean majority agreement (0-1)",
}


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
    total_clocktime: float | None = None
    sample_durations: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)


def strip_log_prefix(line: str) -> str:
    match = INFO_PREFIX_PATTERN.match(line.rstrip("\n"))
    return match.group("message").strip() if match else line.strip()


def stripped_log_text(text: str) -> str:
    return "\n".join(strip_log_prefix(line) for line in text.splitlines())


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_float_list(value: str) -> list[float]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [float(item) for item in parsed if isinstance(item, (int, float))]


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

    text = path.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = stripped_log_text(text)

    total_clocktime_match = TOTAL_CLOCKTIME_PATTERN.search(cleaned_text)
    if total_clocktime_match:
        parsed.total_clocktime = float(total_clocktime_match.group("value"))

    durations_match = DURATIONS_PATTERN.search(cleaned_text)
    if durations_match:
        parsed.sample_durations = parse_float_list(durations_match.group("values"))

    quality_match = QUALITY_CORRECT_PATTERN.search(cleaned_text)
    if quality_match:
        parsed.quality_scores = parse_float_list(quality_match.group("values"))

    for raw_line in cleaned_text.splitlines():
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


def solved_threshold(benchmark: str | None) -> float:
    return SOLVED_THRESHOLDS.get(str(benchmark), 1.0)


def quality_solution_metrics(logs: list[MethodLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        if not log.quality_scores and not log.sample_durations and log.total_clocktime is None:
            continue

        threshold = solved_threshold(log.benchmark)
        solved_flags = [score >= threshold for score in log.quality_scores]
        solved_count = sum(solved_flags)
        sample_count = max(len(log.quality_scores), len(log.sample_durations))
        paired_durations = list(zip(log.sample_durations, solved_flags))
        solved_durations = [duration for duration, solved in paired_durations if solved]
        total_sample_duration = sum(log.sample_durations) if log.sample_durations else None

        rows.append(
            {
                "model": log.model,
                "benchmark": log.benchmark,
                "method": log.method,
                "split": log.split,
                "repeat": log.repeat,
                "solved_threshold": threshold,
                "sample_count": sample_count or None,
                "quality_mean": mean(log.quality_scores) if log.quality_scores else None,
                "quality_median": median(log.quality_scores) if log.quality_scores else None,
                "quality_min": min(log.quality_scores) if log.quality_scores else None,
                "quality_max": max(log.quality_scores) if log.quality_scores else None,
                "solved_count": solved_count if log.quality_scores else None,
                "solved_rate": safe_ratio(solved_count, len(log.quality_scores)),
                "total_clocktime": log.total_clocktime,
                "total_sample_duration": total_sample_duration,
                "mean_solution_time": mean(log.sample_durations) if log.sample_durations else None,
                "median_solution_time": median(log.sample_durations) if log.sample_durations else None,
                "mean_solved_solution_time": mean(solved_durations) if solved_durations else None,
                "median_solved_solution_time": median(solved_durations) if solved_durations else None,
                "clocktime_per_solved": safe_ratio(log.total_clocktime, solved_count),
                "sample_time_per_solved": safe_ratio(total_sample_duration, solved_count),
            }
        )
    return pd.DataFrame(rows)


def quality_sample_metrics(logs: list[MethodLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        if not log.quality_scores:
            continue
        threshold = solved_threshold(log.benchmark)
        for sample_index, score in enumerate(log.quality_scores):
            duration = log.sample_durations[sample_index] if sample_index < len(log.sample_durations) else None
            rows.append(
                {
                    "model": log.model,
                    "benchmark": log.benchmark,
                    "method": log.method,
                    "split": log.split,
                    "repeat": log.repeat,
                    "sample": sample_index + 1,
                    "score": score,
                    "duration": duration,
                    "solved_threshold": threshold,
                    "solved": float(score >= threshold),
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
MAX_EDIT_DISTANCE_RESPONSES = 6
MAX_EDIT_DISTANCE_CHARS = 200


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
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def sample_texts_for_distance(texts: list[str], limit: int = MAX_EDIT_DISTANCE_RESPONSES) -> list[str]:
    if len(texts) <= limit:
        return texts
    # Keep the sample deterministic and spread across the full response set.
    indices = {round(i * (len(texts) - 1) / (limit - 1)) for i in range(limit)}
    return [texts[index] for index in sorted(indices)]


def approximate_normalized_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    denom = max(len(a), len(b), 1)
    overlap = min(len(a), len(b))
    mismatches = sum(1 for i in range(overlap) if a[i] != b[i])
    mismatches += abs(len(a) - len(b))
    return mismatches / denom


def mean_pairwise_edit_distance(texts: list[str]) -> float | None:
    if len(texts) < 2:
        return None
    texts = [text[:MAX_EDIT_DISTANCE_CHARS] for text in sample_texts_for_distance(texts)]
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            distances.append(approximate_normalized_distance(texts[i], texts[j]))
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


def benchmark_groups(df: pd.DataFrame):
    return list(df.groupby("benchmark", dropna=False))


def grouped_mean(df: pd.DataFrame, metric: str, group_by: list[str]) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    group_columns = [column for column in group_by if column in df.columns]
    if not group_columns:
        return pd.DataFrame()
    return (
        df.dropna(subset=[metric])
        .groupby(group_columns, dropna=False)[metric]
        .mean()
        .reset_index()
    )


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
        axis.bar(labels, benchmark_df[metric], color=bar_colors(benchmark_df))
        axis.set_title(str(benchmark), fontsize=9)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
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
        plt.bar(labels, benchmark_df[metric], color=bar_colors(benchmark_df))
        add_method_highlight_legend(plt.gca(), benchmark_df)
        plt.title(f"{title} - {benchmark}")
        plt.ylabel(y_label)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(benchmark_path, dpi=180)
        plt.close()


def plot_convergence_by_benchmark(df: pd.DataFrame, output: Path) -> None:
    if df.empty or "trajectory" not in df.columns:
        return
    plt = ensure_matplotlib()
    label_columns = [column for column in ("method",) if column in df.columns]
    if "benchmark" not in df.columns or not label_columns:
        return

    groups = benchmark_groups(df)
    rows, columns = subplot_grid_size(len(groups))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False)
    has_overview_curve = False
    for axis, (benchmark, benchmark_df) in zip(axes.flat, groups):
        has_curve = False
        for group_key, group in benchmark_df.groupby(label_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label = "/".join(str(value) for value in group_key)
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
            axis.plot(range(len(y)), y, marker="o", markersize=2, linewidth=1, label=label)
            has_curve = True
            has_overview_curve = True
        axis.set_title(str(benchmark), fontsize=9)
        axis.set_xlabel("logged step position (index)", fontsize=8)
        axis.set_ylabel("running best score", fontsize=8)
        axis.tick_params(labelsize=7)
        if has_curve:
            axis.legend(fontsize=5, loc="best")
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    if has_overview_curve:
        fig.suptitle("Logged Best-Score Convergence", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    plt.close(fig)

    for benchmark, benchmark_df in groups:
        plt.figure(figsize=(12, 6))
        has_curve = False
        for group_key, group in benchmark_df.groupby(label_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            label = "/".join(str(value) for value in group_key)
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
            plt.plot(range(len(y)), y, marker="o", label=label)
            has_curve = True
        if not has_curve:
            plt.close()
            continue
        plt.title(f"Logged Best-Score Convergence - {benchmark}")
        plt.xlabel("Logged step position (index)")
        plt.ylabel("Running best logged score")
        plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.subplots_adjust(right=0.78)
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(benchmark_path, dpi=180, bbox_inches="tight")
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
    plt.xlabel("majority agreement (0-1)")
    plt.ylabel("mean pairwise edit distance (0-1)")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def plot_raw_agreement_by_method_and_benchmark(df: pd.DataFrame, output: Path) -> None:
    if df.empty or "method" not in df.columns:
        return
    label_columns = [column for column in ("method",) if column in df.columns]
    group_columns = [column for column in ("benchmark", *label_columns) if column in df.columns]
    plot_df = (
        df.dropna(subset=["majority_agreement"])
        .groupby(group_columns, dropna=False)["majority_agreement"]
        .mean()
        .reset_index()
    )
    if plot_df.empty or "benchmark" not in plot_df.columns:
        return
    plt = ensure_matplotlib()
    groups = benchmark_groups(plot_df)
    rows, columns = subplot_grid_size(len(groups))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False)
    for axis, (benchmark, benchmark_df) in zip(axes.flat, groups):
        labels = [
            "\n".join(str(getattr(row, column)) for column in label_columns if hasattr(row, column))
            for row in benchmark_df.itertuples()
        ]
        axis.bar(labels, benchmark_df["majority_agreement"], color=bar_colors(benchmark_df))
        axis.set_title(str(benchmark), fontsize=9)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(metric_axis_label("majority_agreement"), fontsize=8)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        add_method_highlight_legend(axis, benchmark_df)
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    fig.suptitle("Raw Call Majority Agreement", fontsize=13)
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
        plt.bar(labels, benchmark_df["majority_agreement"], color=bar_colors(benchmark_df))
        add_method_highlight_legend(plt.gca(), benchmark_df)
        plt.title(f"Raw Call Majority Agreement - {benchmark}")
        plt.ylabel(metric_axis_label("majority_agreement"))
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(benchmark_path, dpi=180)
        plt.close()


def methods_heatmap_matrix(df: pd.DataFrame, benchmark: Any) -> pd.DataFrame:
    benchmark_df = df[df["benchmark"].eq(benchmark)].dropna(subset=["solved"])
    if benchmark_df.empty:
        return pd.DataFrame()
    plot_df = (
        benchmark_df.groupby(["sample", "method"], dropna=False)["solved"]
        .mean()
        .reset_index()
        .pivot(index="sample", columns="method", values="solved")
        .sort_index()
    )
    return plot_df.reindex(sorted(plot_df.columns), axis=1)


def draw_methods_heatmap(axis, matrix: pd.DataFrame, title: str, show_labels: bool = True):
    image = axis.imshow(matrix.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_title(title, fontsize=9 if show_labels else 8)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7 if show_labels else 5)

    row_count = len(matrix.index)
    if row_count <= 20:
        y_positions = list(range(row_count))
    else:
        step = math.ceil(row_count / 10)
        y_positions = list(range(0, row_count, step))
    axis.set_yticks(y_positions)
    axis.set_yticklabels([f"Run {int(matrix.index[position])}" for position in y_positions], fontsize=7 if show_labels else 5)
    axis.tick_params(which="both", bottom=True, left=True)

    if show_labels:
        axis.set_xlabel("Method", fontsize=8)
        axis.set_ylabel("Sample index (instance)", fontsize=8)

    if row_count <= 60 and len(matrix.columns) <= 20:
        axis.set_xticks([index - 0.5 for index in range(1, len(matrix.columns))], minor=True)
        axis.set_yticks([index - 0.5 for index in range(1, row_count)], minor=True)
        axis.grid(which="minor", axis="both", color="white", linewidth=0.4)
        axis.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_methods_heatmap_by_benchmark(df: pd.DataFrame, output: Path) -> None:
    if df.empty or not {"benchmark", "method", "sample", "solved"}.issubset(df.columns):
        return
    plt = ensure_matplotlib()
    benchmarks = sorted(df["benchmark"].dropna().unique())
    matrices = [(benchmark, methods_heatmap_matrix(df, benchmark)) for benchmark in benchmarks]
    matrices = [(benchmark, matrix) for benchmark, matrix in matrices if not matrix.empty]
    if not matrices:
        return

    rows, columns = subplot_grid_size(len(matrices))
    fig, axes = plt.subplots(rows, columns, figsize=a4_landscape_size(rows), squeeze=False, constrained_layout=True)
    last_image = None
    for axis, (benchmark, matrix) in zip(axes.flat, matrices):
        last_image = draw_methods_heatmap(axis, matrix, str(benchmark), show_labels=False)
    for axis in list(axes.flat)[len(matrices):]:
        axis.axis("off")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), ticks=[0, 1], shrink=0.72)
        cbar.set_ticklabels(["Not solved", "Solved"])
        cbar.set_label("Solved status (0/1)")
    fig.suptitle("Methods Solved Heatmap", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    for benchmark, matrix in matrices:
        fig, axis = plt.subplots(figsize=(max(8, len(matrix.columns) * 0.7), max(6, len(matrix.index) * 0.16)))
        image = draw_methods_heatmap(axis, matrix, f"Methods Solved Heatmap - {benchmark}")
        cbar = fig.colorbar(image, ax=axis, label="Solved status (0/1)", ticks=[0, 1], orientation="vertical")
        cbar.set_ticklabels(["Not solved", "Solved"])
        fig.tight_layout()
        benchmark_path = benchmark_output(output, benchmark)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(benchmark_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


METHOD_GROUP_COLUMNS = ["model", "benchmark", "method"]


def numeric_sum(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.sum()) if not values.empty else None


def numeric_mean(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.mean()) if not values.empty else None


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    data = pd.DataFrame({"value": values, "weight": weights}).dropna()
    data = data[data["weight"] > 0]
    if data.empty:
        return None
    return float((data["value"] * data["weight"]).sum() / data["weight"].sum())


def grouped_method_rows(df: pd.DataFrame):
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    if df.empty or not group_columns:
        return []
    return df.groupby(group_columns, dropna=False)


def base_method_row(group_key: Any, group_columns: list[str], group: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    row = dict(zip(group_columns, group_key))
    row["log_count"] = len(group)
    return row


def aggregate_effort_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    for group_key, group in grouped_method_rows(df):
        row = base_method_row(group_key, group_columns, group)
        accumulated_score = numeric_sum(group["final_score"])
        method_effort = numeric_sum(group["method_effort"])
        calls_total = numeric_sum(group["calls_total"])
        row.update(
            {
                "final_score": numeric_mean(group["final_score"]),
                "accumulated_score": accumulated_score,
                "event_count": numeric_sum(group["event_count"]),
                "method_effort": method_effort,
                "steps_observed": numeric_mean(group["steps_observed"]),
                "steps_to_first_solution": numeric_mean(group["steps_to_first_solution"]),
                "min_steps_to_first_solution": group["steps_to_first_solution"].dropna().min()
                if group["steps_to_first_solution"].notna().any()
                else None,
                "calls_total": calls_total,
                "score_per_method_effort": safe_ratio(accumulated_score, method_effort),
                "score_per_call": safe_ratio(accumulated_score, calls_total),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_usage_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    sum_columns = [
        "calls_total",
        "calls_saved_by_cacher",
        "calls_saved_by_deduplicator",
        "calls_without_cache_or_dedup",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "total_tokens",
        "input_tokens_saved_by_cacher",
        "output_tokens_saved_by_cacher",
        "tokens_saved_by_cacher",
        "input_tokens_saved_by_deduplicator",
        "output_tokens_saved_by_deduplicator",
        "tokens_saved_by_deduplicator",
        "total_tokens_with_saved",
        "input_cost",
        "output_cost",
        "total_cost",
        "input_cost_saved_by_cacher",
        "output_cost_saved_by_cacher",
        "total_cost_saved_by_cacher",
        "input_cost_saved_by_deduplicator",
        "output_cost_saved_by_deduplicator",
        "total_cost_saved_by_deduplicator",
        "total_cost_with_saved",
    ]
    for group_key, group in grouped_method_rows(df):
        row = base_method_row(group_key, group_columns, group)
        accumulated_score = numeric_sum(group["final_score"])
        row["final_score"] = numeric_mean(group["final_score"])
        row["accumulated_score"] = accumulated_score
        for column in sum_columns:
            if column in group:
                row[column] = numeric_sum(group[column])
        saved_tokens = add_optional_values(row.get("tokens_saved_by_cacher"), row.get("tokens_saved_by_deduplicator"))
        saved_cost = add_optional_values(
            row.get("total_cost_saved_by_cacher"),
            row.get("total_cost_saved_by_deduplicator"),
        )
        row["token_savings_ratio"] = safe_ratio(saved_tokens, row.get("total_tokens_with_saved"))
        row["cost_savings_ratio"] = safe_ratio(saved_cost, row.get("total_cost_with_saved"))
        row["score_per_call"] = safe_ratio(accumulated_score, row.get("calls_total"))
        row["score_per_1k_tokens"] = safe_ratio(
            accumulated_score,
            row["total_tokens"] / 1000 if row.get("total_tokens") else None,
        )
        row["score_per_dollar"] = safe_ratio(accumulated_score, row.get("total_cost"))
        row["cost_per_call"] = safe_ratio(row.get("total_cost"), row.get("calls_total"))
        row["tokens_per_call"] = safe_ratio(row.get("total_tokens"), row.get("calls_total"))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_quality_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    for group_key, group in grouped_method_rows(df):
        row = base_method_row(group_key, group_columns, group)
        sample_count = numeric_sum(group["sample_count"])
        solved_count = numeric_sum(group["solved_count"])
        total_clocktime = numeric_sum(group["total_clocktime"])
        total_sample_duration = numeric_sum(group["total_sample_duration"])
        row.update(
            {
                "solved_threshold": numeric_mean(group["solved_threshold"]),
                "sample_count": sample_count,
                "quality_mean": weighted_mean(group["quality_mean"], group["sample_count"]),
                "quality_median": numeric_mean(group["quality_median"]),
                "quality_min": group["quality_min"].dropna().min() if group["quality_min"].notna().any() else None,
                "quality_max": group["quality_max"].dropna().max() if group["quality_max"].notna().any() else None,
                "solved_count": solved_count,
                "solved_rate": safe_ratio(solved_count, sample_count),
                "total_clocktime": total_clocktime,
                "total_sample_duration": total_sample_duration,
                "mean_solution_time": weighted_mean(group["mean_solution_time"], group["sample_count"]),
                "median_solution_time": numeric_mean(group["median_solution_time"]),
                "mean_solved_solution_time": weighted_mean(group["mean_solved_solution_time"], group["solved_count"]),
                "median_solved_solution_time": numeric_mean(group["median_solved_solution_time"]),
                "clocktime_per_solved": safe_ratio(total_clocktime, solved_count),
                "sample_time_per_solved": safe_ratio(total_sample_duration, solved_count),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_diversity_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    for group_key, group in grouped_method_rows(df):
        row = base_method_row(group_key, group_columns, group)
        row.update(
            {
                "action_entropy": numeric_mean(group["action_entropy"]),
                "normalized_action_entropy": numeric_mean(group["normalized_action_entropy"]),
                "mean_unique_action_ratio": numeric_mean(group["mean_unique_action_ratio"]),
                "observed_action_mass": numeric_sum(group["observed_action_mass"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_trajectory(values: pd.Series) -> str:
    by_step: dict[int, list[float]] = defaultdict(list)
    for value in values.dropna():
        try:
            trajectory = json.loads(value)
        except Exception:
            continue
        for point in trajectory:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                step, score = point
                if isinstance(step, int) and isinstance(score, (int, float)):
                    by_step[step].append(float(score))
    return json.dumps([(step, mean(scores)) for step, scores in sorted(by_step.items())])


def aggregate_convergence_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    for group_key, group in grouped_method_rows(df):
        row = base_method_row(group_key, group_columns, group)
        row.update(
            {
                "convergence_auc": numeric_mean(group["convergence_auc"]),
                "final_logged_best": numeric_mean(group["final_logged_best"]),
                "solved_step": numeric_mean(group["solved_step"]),
                "min_solved_step": group["solved_step"].dropna().min() if group["solved_step"].notna().any() else None,
                "trajectory": aggregate_trajectory(group["trajectory"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_raw_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [column for column in METHOD_GROUP_COLUMNS if column in df.columns]
    for group_key, group in grouped_method_rows(df):
        row = base_method_row(group_key, group_columns, group)
        row.update(
            {
                "prompt_count": group["prompt_key"].nunique() if "prompt_key" in group else len(group),
                "response_count": numeric_sum(group["response_count"]),
                "mean_unique_answer_count": numeric_mean(group["unique_answer_count"]),
                "answer_entropy": weighted_mean(group["answer_entropy"], group["response_count"]),
                "normalized_answer_entropy": weighted_mean(group["normalized_answer_entropy"], group["response_count"]),
                "majority_agreement": weighted_mean(group["majority_agreement"], group["response_count"]),
                "mean_pairwise_edit_distance": weighted_mean(group["mean_pairwise_edit_distance"], group["response_count"]),
                "mean_response_length": weighted_mean(group["mean_response_length"], group["response_count"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(logs_dir: Path, raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_logs = iter_method_logs(logs_dir)

    effort = aggregate_effort_by_method(effort_to_solution(method_logs))
    diversity = aggregate_diversity_by_method(exploration_diversity(method_logs))
    convergence = aggregate_convergence_by_method(convergence_auc(method_logs))
    usage = aggregate_usage_by_method(usage_metrics(method_logs))
    quality = aggregate_quality_by_method(quality_solution_metrics(method_logs))
    quality_samples = quality_sample_metrics(method_logs)
    raw = aggregate_raw_by_method(raw_response_consistency(raw_dir))

    effort.to_csv(metric_data(output_dir, "effort_to_solution"), index=False)
    diversity.to_csv(metric_data(output_dir, "exploration_diversity"), index=False)
    convergence.to_csv(metric_data(output_dir, "convergence_auc"), index=False)
    usage.to_csv(metric_data(output_dir, "usage_metrics"), index=False)
    quality.to_csv(metric_data(output_dir, "quality_solution_metrics"), index=False)
    quality_samples.to_csv(metric_data(output_dir, "methods_heatmap"), index=False)
    raw.to_csv(metric_data(output_dir, "raw_response_consistency"), index=False)

    plot_bar_by_benchmark(
        effort,
        "score_per_method_effort",
        "Score per Method Effort",
        metric_plot(output_dir, "score_per_method_effort"),
    )
    plot_bar_by_benchmark(
        diversity,
        "normalized_action_entropy",
        "Exploration Diversity",
        metric_plot(output_dir, "normalized_action_entropy"),
    )
    plot_convergence_by_benchmark(convergence, metric_plot(output_dir, "convergence_auc"))
    plot_bar_by_benchmark(usage, "calls_total", "Total Calls", metric_plot(output_dir, "calls_total"))
    plot_bar_by_benchmark(usage, "total_tokens", "Total Tokens", metric_plot(output_dir, "total_tokens"))
    plot_bar_by_benchmark(usage, "total_cost", "Total Cost", metric_plot(output_dir, "total_cost"))
    plot_bar_by_benchmark(usage, "score_per_dollar", "Score per Dollar", metric_plot(output_dir, "score_per_dollar"))
    plot_bar_by_benchmark(quality, "quality_mean", "Quality Mean", metric_plot(output_dir, "quality_mean"))
    plot_bar_by_benchmark(quality, "solved_rate", "Solved Rate", metric_plot(output_dir, "solved_rate"))
    plot_bar_by_benchmark(quality, "mean_solution_time", "Mean Solution Time", metric_plot(output_dir, "mean_solution_time"))
    plot_bar_by_benchmark(
        quality,
        "mean_solved_solution_time",
        "Mean Solved Solution Time",
        metric_plot(output_dir, "mean_solved_solution_time"),
    )
    plot_bar_by_benchmark(
        quality,
        "clocktime_per_solved",
        "Clocktime per Solved",
        metric_plot(output_dir, "clocktime_per_solved"),
    )
    plot_methods_heatmap_by_benchmark(quality_samples, metric_plot(output_dir, "methods_heatmap"))
    plot_raw_consistency(raw, metric_plot(output_dir, "raw_response_consistency"))
    plot_raw_agreement_by_method_and_benchmark(raw, metric_plot(output_dir, "raw_majority_agreement"))


def main() -> None:
    config = OmegaConf.load("actions_config.yaml")
    write_outputs(
        logs_dir=Path(config.actions.visualize.log_path),
        raw_dir=Path(config.actions.visualize.raw_path),
        output_dir=Path(config.actions.visualize.output_path),
    )


if __name__ == "__main__":
    main()
