import argparse
import importlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable
from omegaconf import OmegaConf

import pandas as pd


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    benchmark_class: str
    path: str
    extractor: Callable[[Any], tuple[str, str]]
    helm_loader: str | None = None
    helm_splitter: str | None = None
    helm_dataset_dir: str | None = None


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


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


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_answer_label(answer: str) -> str:
    cleaned = safe_str(answer).strip()
    if not cleaned:
        return "<empty>"
    upper = cleaned.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper
    if cleaned.lower() in {"yes", "no", "maybe"}:
        return cleaned.lower()
    if len(cleaned) <= 32:
        return cleaned.lower()
    return "<free_text>"


def extract_game24(row: tuple) -> tuple[str, str]:
    return safe_str(row[1]), "24"


def extract_hle(row: tuple) -> tuple[str, str]:
    return safe_str(row[2]), safe_str(row[5])


def extract_hotpotqa(row: tuple) -> tuple[str, str]:
    return safe_str(row[1]), safe_str(row[2])


def extract_humaneval(row: tuple) -> tuple[str, str]:
    return safe_str(row[1]), safe_str(row[3])


def extract_scibench(row: tuple) -> tuple[str, str]:
    return safe_str(row[1]), safe_str(row[2])


def extract_sonnetwriting(row: tuple) -> tuple[str, str]:
    return safe_str(row[1]), safe_str(row[2])


def extract_logiqa(row: tuple) -> tuple[str, str]:
    context = safe_str(row[2])
    question = safe_str(row[3])
    options = "\n".join(
        f"{label}. {safe_str(value)}"
        for label, value in zip(["A", "B", "C", "D"], row[4:8])
    )
    return f"{context}\n{question}\n{options}", safe_str(row[1])


def extract_matharena(row: tuple) -> tuple[str, str]:
    return safe_str(row[1]), safe_str(row[2])


def extract_mapping(row: dict) -> tuple[str, str]:
    input_text = (
        row.get("puzzle")
        or row.get("prompt_text")
        or row.get("instance_input")
        or row.get("question")
        or row.get("findings")
        or row.get("note_text")
        or ""
    )
    answer = row.get("answer") or row.get("target") or ""
    return safe_str(input_text), safe_str(answer)


DATASETS = [
    DatasetSpec("game24", "src.tasks.game24.benchmark.BenchmarkGame24", "datasets/dataset_game24.csv.gz", extract_game24),
    DatasetSpec("hle", "src.tasks.hle.benchmark.BenchmarkHLE", "datasets/dataset_hle.jsonl.gz", extract_hle),
    DatasetSpec("hotpotqa", "src.tasks.hotpotqa.benchmark.BenchmarkHotpotQA", "datasets/dataset_hotpotqa.csv.gz", extract_hotpotqa),
    DatasetSpec("humaneval", "src.tasks.humaneval.benchmark.BenchmarkHumanEval", "datasets/dataset_humaneval.csv.gz", extract_humaneval),
    DatasetSpec("scibench", "src.tasks.scibench.benchmark.BenchmarkSciBench", "datasets/dataset_scibench.csv.gz", extract_scibench),
    DatasetSpec("sonnetwriting", "src.tasks.sonnetwriting.benchmark.BenchmarkSonnetWriting", "datasets/dataset_sonnetwriting.jsonl.gz", extract_sonnetwriting),
    DatasetSpec("logiqa", "src.tasks.logiqa.benchmark.BenchmarkLogiQA", "datasets/dataset_logiqa.csv.gz", extract_logiqa),
    DatasetSpec("matharena", "src.tasks.matharena.benchmark.BenchmarkMathArena", "datasets/dataset_matharena.jsonl.gz", extract_matharena),
    DatasetSpec(
        "pubmed_qa",
        "src.tasks.pubmed_qa.benchmark.BenchmarkPubMedQA",
        "datasets/dataset_pubmed_qa.csv.gz",
        extract_mapping,
        helm_loader="src.tasks.pubmed_qa.benchmark.load_instances",
        helm_splitter="src.tasks.pubmed_qa.benchmark.split_instances",
        helm_dataset_dir="datasets/medical/pubmed_qa",
    ),
    DatasetSpec(
        "mimic_rrs",
        "src.tasks.mimic_rrs.benchmark.BenchmarkMimic_RRS",
        "datasets/dataset_mimic_rrs.csv.gz",
        extract_mapping,
        helm_loader="src.tasks.mimic_rrs.benchmark.load_instances",
        helm_splitter="src.tasks.mimic_rrs.benchmark.split_instances",
        helm_dataset_dir="datasets/medical/mimic_rrs",
    ),
    DatasetSpec(
        "mtsamples_procedures",
        "src.tasks.mtsamples_procedures.benchmark.BenchmarkMTSamplesProcedures",
        "datasets/dataset_mtsamples_procedures.csv.gz",
        extract_mapping,
        helm_loader="src.tasks.mtsamples_procedures.benchmark.load_instances",
        helm_splitter="src.tasks.mtsamples_procedures.benchmark.split_instances",
        helm_dataset_dir="datasets/medical/mtsamples_procedures",
    ),
]


def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_helm_dataset(spec: DatasetSpec, split: str) -> list[dict[str, Any]]:
    if not spec.helm_loader or not spec.helm_splitter or not spec.helm_dataset_dir:
        raise ValueError(f"Dataset {spec.name} is missing HELM loader configuration.")

    loader = import_class(spec.helm_loader)
    splitter = import_class(spec.helm_splitter)
    instances = loader(Path(spec.helm_dataset_dir))
    return splitter(instances, split)


def split_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "full":
        return records
    if split == "single":
        return records[:1]
    if split == "mini":
        return records[:10]
    if split.startswith("n["):
        match = re.fullmatch(r"n\[(\d+)\]", split)
        return records[: int(match.group(1))] if match else records

    shuffled = list(records)
    import random

    random.Random(0).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * 0.6)
    n_validation = int(n_total * 0.2)

    if split == "train":
        return shuffled[:n_train]
    if split == "validation":
        return shuffled[n_train:n_train + n_validation]
    if split == "test":
        return shuffled[n_train + n_validation:]
    raise ValueError(f"Invalid split name: {split}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_local_medical_dataset(spec: DatasetSpec, split: str) -> list[dict[str, Any]] | None:
    if not spec.helm_dataset_dir:
        return None

    dataset_dir = Path(spec.helm_dataset_dir)
    if spec.name == "pubmed_qa":
        json_path = dataset_dir / "ori_pqal.json"
        if not json_path.exists():
            return None
        source = json.loads(json_path.read_text(encoding="utf-8"))
        records = []
        for source_id, item in source.items():
            question = safe_str(item.get("QUESTION")).strip()
            labels = item.get("LABELS") or []
            contexts = item.get("CONTEXTS") or []
            answer = safe_str(item.get("final_decision")).strip().lower()
            if not question or answer not in {"yes", "no", "maybe"}:
                continue
            background = "\n".join(
                f"{safe_str(label).title()}. {safe_str(context)}"
                for label, context in zip(labels, contexts)
            )
            instance_input = f"Context: {background}\n\nQuestion: {question}"
            records.append(
                {
                    "id": safe_str(source_id),
                    "question": question,
                    "instance_input": instance_input,
                    "prompt_text": (
                        "Answer A for yes, B for no or C for maybe.\n\n"
                        f"Question: {instance_input}\n"
                        "A. yes\n"
                        "B. no\n"
                        "C. maybe\n"
                        "Answer:"
                    ),
                    "answer": answer,
                }
            )
        return split_records(records, split)

    if spec.name == "mimic_rrs":
        findings_path = dataset_dir / "test.findings.tok"
        impression_path = dataset_dir / "test.impression.tok"
        if not findings_path.exists() or not impression_path.exists():
            return None
        findings = findings_path.read_text(encoding="utf-8", errors="replace").splitlines()
        impressions = impression_path.read_text(encoding="utf-8", errors="replace").splitlines()
        records = [
            {
                "id": f"mimic_rrs_{idx}",
                "findings": finding.strip(),
                "answer": impression.strip(),
            }
            for idx, (finding, impression) in enumerate(zip(findings, impressions))
            if finding.strip() and impression.strip()
        ]
        return split_records(records, split)

    if spec.name == "mtsamples_procedures":
        if not dataset_dir.exists():
            return None
        records = []
        for path in sorted(dataset_dir.glob("*.txt")):
            note_text = read_text(path)
            if not note_text:
                continue
            records.append(
                {
                    "id": path.stem,
                    "title": path.stem,
                    "note_text": note_text,
                    "answer": path.stem,
                }
            )
        return split_records(records, split)

    return None


def load_dataset(spec: DatasetSpec, split: str) -> tuple[pd.DataFrame, str | None]:
    try:
        if spec.helm_loader:
            try:
                data = load_helm_dataset(spec, split)
            except Exception:
                local_data = load_local_medical_dataset(spec, split)
                if local_data is None:
                    raise
                data = local_data
        else:
            benchmark_cls = import_class(spec.benchmark_class)
            benchmark = benchmark_cls(spec.path, split=split)
            data = getattr(benchmark, "data", [])
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"

    rows = []
    for i, row in enumerate(data):
        try:
            input_text, answer_text = spec.extractor(row)
        except Exception:
            input_text, answer_text = "", ""
        input_tokens = tokens(input_text)
        answer_tokens = tokens(answer_text)
        rows.append(
            {
                "dataset": spec.name,
                "sample_idx": i,
                "input_text": input_text,
                "answer_text": answer_text,
                "input_word_count": len(input_tokens),
                "answer_word_count": len(answer_tokens),
                "input_char_count": len(input_text),
                "answer_char_count": len(answer_text),
                "label": normalize_answer_label(answer_text),
            }
        )
    return pd.DataFrame(rows), None


def corpus_lexical_diversity(texts: list[str]) -> float:
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokens(text))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def corpus_input_stats(texts: list[str]) -> dict[str, float | int]:
    all_tokens = []
    sentence_lengths = []
    for text in texts:
        text_tokens = tokens(text)
        all_tokens.extend(text_tokens)
        for sentence in SENTENCE_PATTERN.findall(str(text)):
            length = len(tokens(sentence))
            if length:
                sentence_lengths.append(length)

    if not all_tokens:
        return {
            "input_vocab_size": 0,
            "input_hapax_share": 0.0,
            "mean_input_sentence_words": 0.0,
            "input_stopword_share": 0.0,
        }

    token_counts = Counter(all_tokens)
    return {
        "input_vocab_size": len(token_counts),
        "input_hapax_share": sum(count == 1 for count in token_counts.values()) / len(token_counts),
        "mean_input_sentence_words": mean(sentence_lengths) if sentence_lengths else 0.0,
        "input_stopword_share": sum(token in STOPWORDS for token in all_tokens) / len(all_tokens),
    }


def aggregate_metrics(samples: pd.DataFrame, failures: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    if not samples.empty:
        for dataset, group in samples.groupby("dataset"):
            label_counts = Counter(group["label"])
            input_stats = corpus_input_stats(group["input_text"].tolist())
            rows.append(
                {
                    "dataset": dataset,
                    "status": "loaded",
                    "num_instances": len(group),
                    "median_input_words": median(group["input_word_count"]),
                    "mean_input_words": mean(group["input_word_count"]),
                    "median_answer_words": median(group["answer_word_count"]),
                    "mean_answer_words": mean(group["answer_word_count"]),
                    "input_lexical_diversity": corpus_lexical_diversity(group["input_text"].tolist()),
                    **input_stats,
                    "answer_entropy": entropy(label_counts),
                    "normalized_answer_entropy": normalized_entropy(label_counts),
                    "dominant_label_share": max(label_counts.values()) / len(group) if len(group) else None,
                    "free_text_answer_share": sum(group["label"] == "<free_text>") / len(group) if len(group) else None,
                }
            )

    for failure in failures:
        rows.append(
            {
                "dataset": failure["dataset"],
                "status": failure["error"],
                "num_instances": 0,
                "median_input_words": None,
                "mean_input_words": None,
                "median_answer_words": None,
                "mean_answer_words": None,
                "input_lexical_diversity": None,
                "input_vocab_size": None,
                "input_hapax_share": None,
                "mean_input_sentence_words": None,
                "input_stopword_share": None,
                "answer_entropy": None,
                "normalized_answer_entropy": None,
                "dominant_label_share": None,
                "free_text_answer_share": None,
            }
        )

    return pd.DataFrame(rows).sort_values("dataset")


def ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with `pip install matplotlib`.") from exc
    return plt


def loaded_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["status"] == "loaded"].copy()


def format_metric_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def annotate_bars(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        ax.annotate(
            format_metric_value(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def annotate_points(ax: Any, x_values: list[Any], y_values: list[float]) -> None:
    for x, y in zip(x_values, y_values):
        if pd.isna(y):
            continue
        ax.annotate(
            format_metric_value(y),
            xy=(x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_dataset_size(metrics: pd.DataFrame, output: Path) -> None:
    df = loaded_metrics(metrics)
    if df.empty:
        return
    plt = ensure_matplotlib()
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df["dataset"], df["num_instances"])
    annotate_bars(plt.gca(), bars)
    plt.title("Dataset Size")
    plt.ylabel("instances")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_input_length(samples: pd.DataFrame, output: Path) -> None:
    if samples.empty:
        return
    plt = ensure_matplotlib()
    datasets = sorted(samples["dataset"].unique())
    data = [samples[samples["dataset"] == dataset]["input_word_count"].tolist() for dataset in datasets]
    plt.figure(figsize=(10, 5))
    try:
        boxplot = plt.boxplot(data, tick_labels=datasets, showfliers=False)
    except TypeError:
        boxplot = plt.boxplot(data, labels=datasets, showfliers=False)
    ax = plt.gca()
    for i, median_line in enumerate(boxplot["medians"], start=1):
        median_value = median_line.get_ydata()[0]
        ax.annotate(
            format_metric_value(median_value),
            xy=(i, median_value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.title("Input Length Distribution")
    plt.ylabel("input words")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_answer_length_vs_entropy(metrics: pd.DataFrame, output: Path) -> None:
    df = loaded_metrics(metrics).dropna(subset=["median_answer_words", "normalized_answer_entropy"])
    if df.empty:
        return
    plt = ensure_matplotlib()
    plt.figure(figsize=(8, 5))
    sizes = [max(30, value * 2) for value in df["num_instances"]]
    plt.scatter(df["median_answer_words"], df["normalized_answer_entropy"], s=sizes, alpha=0.75)
    for row in df.itertuples():
        label = (
            f"{row.dataset}\n"
            f"{format_metric_value(row.median_answer_words)}, "
            f"{format_metric_value(row.normalized_answer_entropy)}"
        )
        plt.annotate(label, (row.median_answer_words, row.normalized_answer_entropy), fontsize=7)
    plt.title("Answer Length vs Answer-Space Entropy")
    plt.xlabel("median answer words")
    plt.ylabel("normalized answer entropy")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_lexical_diversity(metrics: pd.DataFrame, output: Path) -> None:
    df = loaded_metrics(metrics).dropna(subset=["input_lexical_diversity"])
    if df.empty:
        return
    plt = ensure_matplotlib()
    plt.figure(figsize=(10, 5))
    plt.plot(df["dataset"], df["input_lexical_diversity"], marker="o")
    annotate_points(plt.gca(), df["dataset"].tolist(), df["input_lexical_diversity"].tolist())
    plt.title("Input Lexical Diversity")
    plt.ylabel("type-token ratio")
    plt.ylim(0, min(1.05, max(df["input_lexical_diversity"]) + 0.1))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_vocabulary_profile(metrics: pd.DataFrame, output: Path) -> None:
    df = loaded_metrics(metrics).dropna(subset=["input_vocab_size", "input_hapax_share"])
    if df.empty:
        return
    plt = ensure_matplotlib()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(df["dataset"], df["input_vocab_size"], color="#4c78a8")
    annotate_bars(ax1, bars)
    ax1.set_ylabel("unique input tokens")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(df["dataset"], df["input_hapax_share"], color="#f58518", marker="o")
    annotate_points(ax2, df["dataset"].tolist(), df["input_hapax_share"].tolist())
    ax2.set_ylabel("hapax share")
    ax2.set_ylim(0, min(1.05, max(df["input_hapax_share"]) + 0.1))

    plt.title("Input Vocabulary Profile")
    fig.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_sentence_and_stopword_profile(metrics: pd.DataFrame, output: Path) -> None:
    df = loaded_metrics(metrics).dropna(subset=["mean_input_sentence_words", "input_stopword_share"])
    if df.empty:
        return
    plt = ensure_matplotlib()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(df["dataset"], df["mean_input_sentence_words"], color="#54a24b")
    annotate_bars(ax1, bars)
    ax1.set_ylabel("mean sentence words")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(df["dataset"], df["input_stopword_share"], color="#e45756", marker="o")
    annotate_points(ax2, df["dataset"].tolist(), df["input_stopword_share"].tolist())
    ax2.set_ylabel("stopword share")
    ax2.set_ylim(0, min(1.05, max(df["input_stopword_share"]) + 0.1))

    plt.title("Input Sentence and Function-Word Profile")
    fig.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_complexity_heatmap(metrics: pd.DataFrame, output: Path) -> None:
    df = loaded_metrics(metrics)
    cols = [
        "median_input_words",
        "median_answer_words",
        "input_lexical_diversity",
        "input_hapax_share",
        "mean_input_sentence_words",
        "input_stopword_share",
        "normalized_answer_entropy",
        "dominant_label_share",
    ]
    df = df.dropna(subset=cols)
    if df.empty:
        return
    matrix = df[cols].astype(float)
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min()).replace(0, 1)
    plt = ensure_matplotlib()
    plt.figure(figsize=(9, max(4, len(df) * 0.45)))
    plt.imshow(normalized, aspect="auto", cmap="viridis")
    plt.colorbar(label="min-max normalized")
    plt.yticks(range(len(df)), df["dataset"])
    plt.xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax = plt.gca()
    for row_idx in range(len(df)):
        for col_idx, col in enumerate(cols):
            normalized_value = normalized.iloc[row_idx, col_idx]
            text_color = "white" if normalized_value < 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                format_metric_value(matrix.iloc[row_idx, col_idx]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )
    plt.title("Dataset Complexity Profile")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_outputs(output_dir: Path, split: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_frames = []
    failures = []
    for spec in DATASETS:
        frame, error = load_dataset(spec, split)
        if error:
            failures.append({"dataset": spec.name, "error": error})
        elif not frame.empty:
            sample_frames.append(frame)

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    metrics = aggregate_metrics(samples, failures)

    samples.to_csv(output_dir / "dataset_samples.csv", index=False)
    metrics.to_csv(output_dir / "dataset_metrics.csv", index=False)

    plot_dataset_size(metrics, output_dir / "dataset_size.png")
    plot_input_length(samples, output_dir / "input_length_distribution.png")
    plot_answer_length_vs_entropy(metrics, output_dir / "answer_length_vs_entropy.png")
    plot_lexical_diversity(metrics, output_dir / "lexical_diversity.png")
    plot_vocabulary_profile(metrics, output_dir / "vocabulary_profile.png")
    plot_sentence_and_stopword_profile(metrics, output_dir / "sentence_stopword_profile.png")
    plot_complexity_heatmap(metrics, output_dir / "complexity_heatmap.png")


def main() -> None:
    config = OmegaConf.load("actions_config.yaml")
    write_outputs(
        output_dir=Path(config.actions.visualize.output_path) / "dataset",
        split="full",
    )


if __name__ == "__main__":
    main()
