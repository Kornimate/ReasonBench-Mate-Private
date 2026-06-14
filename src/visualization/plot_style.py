from __future__ import annotations

from collections.abc import Iterable
from typing import Any


METHOD_LABELS: dict[str, str] = {
    "io": "IO",
    "cot": "CoT",
    "cot_sc": "CoT-SC",
    "react": "ReAct",
    "foa": "FoA",
    "heterogeneous_foa": "Heterogeneous FoA",
    "tot_bfs": "ToT-BFS",
    "tot_dfs": "ToT-DFS",
    "got": "GoT",
    "rap": "RAP",
    "reagents": "ReAgents",
    "reagents_v2comp": "ReAgents V2Comp",
    "reagents_v2tour": "ReAgents V2Tour",
}

BENCHMARK_LABELS: dict[str, str] = {
    "game24": "Game24",
    "hle": "HLE",
    "hotpotqa": "HotpotQA",
    "humaneval": "HumanEval",
    "logiqa": "LogiQA",
    "matharena": "MathArena",
    "mimic_rrs": "MIMIC-RRS",
    "mtsamples_procedures": "MTSamples Procedures",
    "pubmed_qa": "PubMedQA",
    "scibench": "SciBench",
    "sonnetwriting": "SonnetWriting",
    "OVERALL_MACRO": "Overall Macro",
}

METHOD_COLORS: dict[str, str] = {
    "io": "#1f77b4",
    "cot": "#ff7f0e",
    "cot_sc": "#2ca02c",
    "react": "#d62728",
    "foa": "#9467bd",
    "heterogeneous_foa": "#8c564b",
    "tot_bfs": "#e377c2",
    "tot_dfs": "#7f7f7f",
    "got": "#bcbd22",
    "rap": "#17becf",
    "reagents": "#00429d",
    "reagents_v2comp": "#93003a",
    "reagents_v2tour": "#006d2c",
}

BENCHMARK_COLORS: dict[str, str] = {
    "game24": "#6a3d9a",
    "hle": "#b15928",
    "hotpotqa": "#1b9e77",
    "humaneval": "#d95f02",
    "logiqa": "#7570b3",
    "matharena": "#e7298a",
    "mimic_rrs": "#66a61e",
    "mtsamples_procedures": "#e6ab02",
    "pubmed_qa": "#a6761d",
    "scibench": "#1f78b4",
    "sonnetwriting": "#b2df8a",
    "OVERALL_MACRO": "#4d4d4d",
}

FALLBACK_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]

MODERN_RC_PARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#111111",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "font.size": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#d9dde3",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.6,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}


def sentence_case(text: object) -> str:
    value = str(text)
    return value[:1].upper() + value[1:] if value else value


def fallback_label(value: Any) -> str:
    text = str(value)
    if not text:
        return "Unknown"
    parts = text.split("_")
    return " ".join(part.upper() if len(part) <= 3 else part.capitalize() for part in parts)


def method_label(method: Any) -> str:
    key = str(method)
    return METHOD_LABELS.get(key, fallback_label(key))


def benchmark_label(benchmark: Any) -> str:
    key = str(benchmark)
    return BENCHMARK_LABELS.get(key, fallback_label(key))


def method_labels(methods: Iterable[Any]) -> list[str]:
    return [method_label(method) for method in methods]


def benchmark_labels(benchmarks: Iterable[Any]) -> list[str]:
    return [benchmark_label(benchmark) for benchmark in benchmarks]


def fallback_color(value: Any) -> str:
    index = sum(ord(char) for char in str(value)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[index]


def method_color(method: Any) -> str:
    key = str(method)
    return METHOD_COLORS.get(key, fallback_color(key))


def benchmark_color(benchmark: Any) -> str:
    key = str(benchmark)
    return BENCHMARK_COLORS.get(key, fallback_color(key))


def method_colors(methods: Iterable[Any]) -> list[str]:
    return [method_color(method) for method in methods]


def benchmark_colors(benchmarks: Iterable[Any]) -> list[str]:
    return [benchmark_color(benchmark) for benchmark in benchmarks]


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(MODERN_RC_PARAMS)


def apply_plot_area_style(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333")
