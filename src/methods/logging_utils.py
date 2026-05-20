import json
import logging
from collections import Counter
from typing import Any, Iterable


logger = logging.getLogger("__main__")


def round_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: round_log_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_log_value(item) for item in value]
    if isinstance(value, tuple):
        return [round_log_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return round_log_value(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return round(value, 4)
    return value


def score_summary(values: Iterable[Any]) -> dict[str, Any]:
    normalized = [normalize_score(value) for value in values]
    if not normalized:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "values": [],
        }
    return {
        "count": len(normalized),
        "min": min(normalized),
        "max": max(normalized),
        "mean": sum(normalized) / len(normalized),
        "values": normalized,
    }


def normalize_score(score: Any) -> float:
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, tuple) and score:
        return normalize_score(score[-1])
    if isinstance(score, list) and score:
        return normalize_score(score[-1])
    return 0.0


def action_summary(actions: Iterable[Any]) -> dict[str, Any]:
    flattened = []
    for action in actions:
        if isinstance(action, list):
            flattened.extend(str(item) for item in action)
        else:
            flattened.append(str(action))

    counts = Counter(flattened)
    return {
        "count": len(flattened),
        "unique": len(counts),
        "most_common": counts.most_common(5),
    }


def state_depths(states: Iterable[Any]) -> list[int | None]:
    depths = []
    for state in states:
        steps = getattr(state, "steps", None)
        depths.append(len(steps) if steps is not None else None)
    return depths


def terminal_summary(env: Any, states: Iterable[Any]) -> dict[str, Any]:
    terminal = 0
    solved = 0
    scores = []
    for state in states:
        try:
            is_terminal = bool(env.is_final(state))
        except Exception:
            is_terminal = False
        try:
            _, score = env.evaluate(state)
            score = normalize_score(score)
        except Exception:
            score = 0.0
        terminal += int(is_terminal)
        solved += int(score == 1.0)
        scores.append(score)
    return {
        "terminal_count": terminal,
        "solved_count": solved,
        "scores": score_summary(scores),
    }


def log_section(title: str) -> None:
    logger.info(title)


def log_event(label: str, payload: dict[str, Any]) -> None:
    logger.info("\t%s %s", label, json.dumps(round_log_value(payload)))


def log_section_end() -> None:
    logger.info("")
