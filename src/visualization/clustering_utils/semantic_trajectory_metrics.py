#!/usr/bin/env python3
"""Conversation-derived semantic trajectory metrics for ReasonBench raw logs.

This script parses raw transcript logs such as mimic_rrs/*.log, extracts source
findings and proposal-generating responses, embeds proposals and findings, and
computes per-case and per-method semantic process metrics.

Recommended production execution for medical text:
    python semantic_trajectory_metrics.py mimic_rrs.zip --backend sentence-transformers \
        --model FremyCompany/BioLORD-2023-M --output-dir results/biolord

A deterministic no-download TF-IDF backend is included for smoke testing only:
    python semantic_trajectory_metrics.py mimic_rrs.zip --backend tfidf --output-dir results/tfidf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEPARATOR_RE = re.compile(r"(?:={100}\n?)+")
RESPONSE_RE = re.compile(
    r"\* RESPONSE \d+ \*\n\*+\n(.*?)(?=(?:\n\*+\n\* RESPONSE \d+ \*\n\*+)|\Z)", re.S
)
N_RE = re.compile(r"\n\*+\n\* N:\s*(\d+)\s*\*\n\*+\n")
SOURCE_BLOCK_PATTERNS = [
    re.compile(r"\bFindings:\s*\n(.*?)(?:\n\n|$)", re.S),
    re.compile(r"\bPatient Notes:\s*(.*?)(?:\n\n(?:Think|Rules|You are|Current draft|Response:|Answer:)|\Z)", re.S),
    re.compile(r"\bInput Problem:\s*(.*?)(?:\n\n|\Z)", re.S),
    re.compile(r"\bQuestion:\s*(.*?)(?:\n\n|\Z)", re.S),
    re.compile(r"\bProblem:\s*(.*?)(?:\n\n|\Z)", re.S),
]
LAST_LINE_SOURCE_PATTERNS = [
    re.compile(r"^\s*Input:\s*(.+?)\s*$", re.M),
]
PROPOSAL_COLUMNS = ["method", "case_id", "call_index", "response_index", "proposal_index", "findings", "proposal"]
RESOURCE_COLUMNS = [
    "method",
    "case_id",
    "raw_calls",
    "raw_completions",
    "unassigned_call_overhead",
    "unassigned_completion_overhead",
]


def _prompt_body(prompt: str) -> str:
    user_blocks = re.findall(r"\*+\n\* USER \*\n\*+\n(.*?)(?=(?:\n\*+\n\* [A-Z]+ \*\n\*+)|\Z)", prompt, re.S)
    if user_blocks:
        return user_blocks[-1].strip()
    return re.sub(r"^\*+\n\* USER \*\n\*+\n", "", prompt).strip()


def _generic_source(body: str) -> str | None:
    body = body.strip()
    if not body:
        return None
    parts = re.split(r"\n\s*\n", body, maxsplit=1)
    if len(parts) == 2 and re.search(r"\b(given|return|select|write|participating|answer)\b", parts[0], re.I):
        body = parts[1].strip()
    body = re.sub(r"\nCurrent (?:answer|draft|reasoning|implementation):\s*\Z", "", body, flags=re.I).strip()
    return body or None


def parse_transcript(text: str) -> list[dict]:
    """Parse one raw LLM transcript into prompt/response calls."""
    text = text.replace("\r\n", "\n")
    calls: list[dict] = []
    for block in (part.strip() for part in SEPARATOR_RE.split(text)):
        if not block:
            continue
        marker = N_RE.search(block)
        if not marker:
            continue
        prompt = block[: marker.start()].strip()
        responses = [item.strip() for item in RESPONSE_RE.findall(block[marker.end() :])]
        calls.append({"prompt": prompt, "n": int(marker.group(1)), "responses": responses})
    return calls


def extract_findings(prompt: str) -> str | None:
    """Extract and normalize the task source text from a prompt."""
    body = _prompt_body(prompt)
    for pattern in SOURCE_BLOCK_PATTERNS:
        matches = [match.group(1).strip() for match in pattern.finditer(body)]
        if matches:
            source = matches[-1]
            source = re.sub(r"\nImpression:\s*$", "", source, flags=re.I).strip()
            return source or None
    for pattern in LAST_LINE_SOURCE_PATTERNS:
        matches = [match.group(1).strip() for match in pattern.finditer(body)]
        if matches:
            return matches[-1] or None
    return _generic_source(body)


def is_proposal_prompt(prompt: str) -> bool:
    """Keep draft-generation calls, excluding evaluator and adaptive-control calls."""
    body = _prompt_body(prompt)
    if extract_findings(prompt) is None:
        return False
    return not _is_excluded_prompt_body(body)


def _is_excluded_prompt_body(body: str) -> bool:
    excluded_markers = [
        "You are evaluating whether",
        "You are the judge",
        "You are judging",
        "You are selecting",
        "Please select",
        "Return only the candidate numbers",
        "Return only the indexes",
        "Selected candidates:",
        "Selected answers:",
        "Selected Next Step Set:",
        "Evaluation:",
        "Evaluation scores:",
        "Score: <number>",
        "Judge:",
    ]
    return any(marker in body for marker in excluded_markers)


def extract_proposals(response: str) -> list[str]:
    """Extract final/proposed impression text while omitting thoughts and scores."""
    response = response.strip()
    final_sections = re.findall(r"Final Section:\s*(.*?)(?=\n[A-Z][A-Za-z ]*:\s*|\Z)", response, re.S)
    if final_sections and final_sections[-1].strip():
        return [final_sections[-1].strip()]
    candidates = re.findall(
        r"(?:^|\n)Candidate\s+\d+:\s*(.*?)(?=(?:\nCandidate\s+\d+:)|\Z)",
        response,
        re.S | re.I,
    )
    if candidates:
        return [candidate.strip() for candidate in candidates if candidate.strip()]
    draft = re.search(r"(?:^|\n)Draft:\s*(.*?)(?=\n(?:Thought|Score|Evaluation):|\Z)", response, re.S | re.I)
    if draft and draft.group(1).strip():
        return [draft.group(1).strip()]
    cleaned = re.sub(r"^(?:Impression:\s*)", "", response, flags=re.I).strip()
    return [cleaned] if cleaned else []


def method_name(path: Path) -> str:
    return re.sub(r"_n\[.*$", "", path.stem)


def unique_method_name(source: Path, path: Path, duplicate_stems: set[str]) -> str:
    name = method_name(path)
    if name not in duplicate_stems:
        return name
    relative = path.relative_to(source).with_suffix("").as_posix()
    return re.sub(r"_n\[.*$", "", relative)


def read_log_files(source: Path) -> dict[str, list[dict]]:
    logs: dict[str, list[dict]] = {}
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if name.endswith(".log")]
            stems = Counter(method_name(Path(name)) for name in names)
            duplicate_stems = {name for name, count in stems.items() if count > 1}
            for name in names:
                path = Path(name)
                key = unique_method_name(Path("."), path, duplicate_stems)
                logs[key] = parse_transcript(archive.read(name).decode("utf-8", errors="replace"))
    else:
        paths = sorted(source.rglob("*.log"))
        stems = Counter(method_name(path) for path in paths)
        duplicate_stems = {name for name, count in stems.items() if count > 1}
        for path in paths:
            key = unique_method_name(source, path, duplicate_stems)
            logs[key] = parse_transcript(path.read_text(encoding="utf-8", errors="replace"))
    if not logs:
        raise ValueError(f"No .log transcript files found under {source}")
    return logs


def proposal_table(logs: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    findings = sorted(
        {
            found
            for calls in logs.values()
            for call in calls
            if not _is_excluded_prompt_body(_prompt_body(call["prompt"]))
            if (found := extract_findings(call["prompt"])) is not None
        }
    )
    case_map = {finding: idx for idx, finding in enumerate(findings)}
    proposal_rows: list[dict] = []
    resource_rows: list[dict] = []
    for method, calls in logs.items():
        resources: defaultdict[int, dict[str, float]] = defaultdict(lambda: {"raw_calls": 0.0, "raw_completions": 0.0})
        unassigned_calls = 0
        unassigned_completions = 0
        for call_index, call in enumerate(calls):
            finding = extract_findings(call["prompt"])
            if finding is None or finding not in case_map:
                unassigned_calls += 1
                unassigned_completions += len(call["responses"])
                continue
            case_id = case_map[finding]
            resources[case_id]["raw_calls"] += 1
            resources[case_id]["raw_completions"] += len(call["responses"])
            if not is_proposal_prompt(call["prompt"]):
                continue
            for response_index, response in enumerate(call["responses"]):
                for proposal_index, proposal in enumerate(extract_proposals(response)):
                    proposal_rows.append(
                        {
                            "method": method,
                            "case_id": case_id,
                            "call_index": call_index,
                            "response_index": response_index,
                            "proposal_index": proposal_index,
                            "findings": finding,
                            "proposal": proposal,
                        }
                    )
        # Adaptive-control prompts in ReAgents contain no findings. Allocate their
        # resource overhead evenly: they count against usage but not semantic score.
        per_case_call_overhead = unassigned_calls / len(findings) if findings else 0.0
        per_case_completion_overhead = unassigned_completions / len(findings) if findings else 0.0
        for case_id in range(len(findings)):
            resource_rows.append(
                {
                    "method": method,
                    "case_id": case_id,
                    "raw_calls": resources[case_id]["raw_calls"] + per_case_call_overhead,
                    "raw_completions": resources[case_id]["raw_completions"] + per_case_completion_overhead,
                    "unassigned_call_overhead": per_case_call_overhead,
                    "unassigned_completion_overhead": per_case_completion_overhead,
                }
            )
    proposals = pd.DataFrame(proposal_rows, columns=PROPOSAL_COLUMNS).sort_values(
        ["method", "case_id", "call_index", "response_index", "proposal_index"]
    )
    resources = pd.DataFrame(resource_rows, columns=RESOURCE_COLUMNS)
    return proposals.reset_index(drop=True), resources, findings


def embed_texts(texts: list[str], backend: str, model: str | None) -> np.ndarray:
    if backend == "tfidf":
        return TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(texts)
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}, only tfidf is implemented")


def pairwise_distance_mean(matrix) -> float | None:
    if matrix.shape[0] < 2:
        return None
    similarities = cosine_similarity(matrix)
    distances = 1.0 - similarities[np.triu_indices(matrix.shape[0], 1)]
    return float(np.mean(distances)) if distances.size else None


def compute_metrics(proposals: pd.DataFrame, resources: pd.DataFrame, findings: list[str], backend: str, model: str | None):
    texts = findings + proposals["proposal"].tolist()
    embedded = embed_texts(texts, backend=backend, model=model)
    finding_vectors = embedded[: len(findings)]
    proposal_vectors = embedded[len(findings) :]
    proposals = proposals.copy()
    proposals["source_alignment"] = [
        float(cosine_similarity(proposal_vectors[i : i + 1], finding_vectors[int(case_id) : int(case_id) + 1])[0, 0])
        for i, case_id in enumerate(proposals["case_id"])
    ]
    case_rows: list[dict] = []
    for (method, case_id), group in proposals.groupby(["method", "case_id"], sort=False):
        indices = group.index.to_numpy()
        vectors = proposal_vectors[indices]
        count = len(group)
        quartile = max(1, math.ceil(count / 4))
        early_diversity = pairwise_distance_mean(vectors[:quartile])
        late_diversity = pairwise_distance_mean(vectors[-quartile:])
        alignments = group["source_alignment"].to_numpy()
        case_rows.append(
            {
                "method": method,
                "case_id": case_id,
                "proposal_count": count,
                "source_alignment_mean": float(np.mean(alignments)),
                "source_alignment_best": float(np.max(alignments)),
                "semantic_diversity": pairwise_distance_mean(vectors),
                "early_diversity": early_diversity,
                "late_consensus": None if late_diversity is None else 1.0 - late_diversity,
                "best_alignment_auc": float(np.mean(np.maximum.accumulate(alignments))),
            }
        )
    cases = pd.DataFrame(case_rows).merge(resources, on=["method", "case_id"], how="left")
    # GEE rewards breadth among early grounded proposals but charges diminishing
    # resource cost. It is meaningful only when at least two early proposals exist.
    cases["gee_proposal_budget"] = (
        cases["source_alignment_best"] * cases["early_diversity"] / np.log1p(cases["proposal_count"])
    )
    cases["gee_full_transcript"] = (
        cases["source_alignment_best"] * cases["early_diversity"] / np.log1p(cases["raw_completions"])
    )
    cases["explore_then_consolidate_grounding"] = (
        cases["source_alignment_best"] * cases["early_diversity"] * cases["late_consensus"]
    )
    numeric = [column for column in cases.select_dtypes(include=["number"]).columns if column != "case_id"]
    methods = cases.groupby("method", as_index=False)[numeric].mean()
    for metric in ["gee_full_transcript", "gee_proposal_budget", "explore_then_consolidate_grounding", "source_alignment_best"]:
        methods[f"rank_{metric}"] = methods[metric].rank(method="min", ascending=False, na_option="bottom")
    return proposals, cases, methods.sort_values(["rank_gee_full_transcript", "method"])


def render_summary(methods: pd.DataFrame, backend: str, model: str | None, case_count: int) -> str:
    model_label = model if backend != "tfidf" else "word+bigram TF-IDF diagnostic baseline"
    lines = [
        "# Semantic trajectory metric results",
        "",
        f"- Cases parsed: {case_count}",
        f"- Vector backend: `{backend}` ({model_label})",
        "- Primary metric: `gee_full_transcript` (Grounded Exploration Efficiency including all logged completions).",
        "- Interpretation warning: semantic source alignment is not a factuality/clinical-correctness score; combine it with benchmark quality before using it as an overall winner metric.",
        "",
        "## Primary ranking: Grounded Exploration Efficiency, full transcript",
        "",
        methods[["method", "gee_full_transcript", "rank_gee_full_transcript", "raw_completions", "source_alignment_best", "early_diversity"]]
        .sort_values("rank_gee_full_transcript")
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Proposal-budget sensitivity ranking",
        "",
        methods[["method", "gee_proposal_budget", "rank_gee_proposal_budget", "proposal_count"]]
        .sort_values("rank_gee_proposal_budget")
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Formula",
        "",
        "For each case, with source findings `f`, generated proposal texts `p_t`, cosine similarity `s`, early-proposal set `E`, and logged completion count `C`:",
        "",
        "`GEE_full = max_t s(f, p_t) * mean_{a<b in E}(1 - s(p_a, p_b)) / ln(1 + C)`",
        "",
        "It rewards grounded early exploration and penalizes methods that spend many logged completions obtaining it.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A ZIP containing .log files or a raw-log directory")
    parser.add_argument("--backend", choices=["tfidf", "sentence-transformers"], default="tfidf")
    parser.add_argument("--model", default=None, help="SentenceTransformer model; default for embedding backend is BioLORD-2023-M")
    parser.add_argument("--output-dir", type=Path, default=Path("semantic_metric_results"))
    args = parser.parse_args()

    logs = read_log_files(args.source)
    proposals, resources, findings = proposal_table(logs)
    proposals, cases, methods = compute_metrics(proposals, resources, findings, args.backend, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(args.output_dir / "proposals.csv", index=False)
    cases.to_csv(args.output_dir / "metrics_by_case.csv", index=False)
    methods.to_csv(args.output_dir / "metrics_by_method.csv", index=False)
    (args.output_dir / "summary.md").write_text(render_summary(methods, args.backend, args.model, len(findings)), encoding="utf-8")
    print(methods[["method", "gee_full_transcript", "rank_gee_full_transcript", "gee_proposal_budget", "rank_gee_proposal_budget"]].to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
