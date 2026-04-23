import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


POSSIBLE_ANSWER_CHOICES: List[str] = ["yes", "no", "maybe"]
LETTER_TO_ANSWER: Dict[str, str] = {"A": "yes", "B": "no", "C": "maybe"}

INSTRUCTIONS = (
    "Answer A for yes, B for no or C for maybe. "
    "Do not include any explanation or additional text. "
    "Output only the letter on a single line."
)


@dataclass
class PubMedQAItem:
    question: str
    labels: List[str]
    contexts: List[str]
    gold_answer: str


def normalize_text(text: str, should_remove_articles: bool = True) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    normalized_text = remove_punc(text.lower())
    if should_remove_articles:
        normalized_text = remove_articles(normalized_text)
    return white_space_fix(normalized_text)


def exact_match(gold: str, pred: Optional[str]) -> float:
    if not pred:
        return 0.0
    return 1.0 if gold.strip() == pred.strip() else 0.0


def quasi_exact_match(gold: str, pred: Optional[str]) -> float:
    if not pred:
        return 0.0
    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0


def prefix_exact_match(gold: str, pred: Optional[str]) -> float:
    if not pred:
        return 0.0
    return 1.0 if pred.strip().startswith(gold.strip()) else 0.0


def quasi_prefix_exact_match(gold: str, pred: Optional[str]) -> float:
    if not pred:
        return 0.0
    return 1.0 if normalize_text(pred).startswith(normalize_text(gold)) else 0.0


def build_background(labels: List[str], contexts: List[str]) -> str:
    return "\n".join(f"{label.title()}. {context}" for label, context in zip(labels, contexts))


def build_instance_input(item: PubMedQAItem) -> str:
    background = build_background(item.labels, item.contexts)
    return f"Context: {background}\n\nQuestion: {item.question}\n"


def build_helm_prompt(item: PubMedQAItem) -> str:
    instance_input = build_instance_input(item)
    return (
        f"{INSTRUCTIONS}\n\n"
        f"Question: {instance_input}\n"
        f"A. yes\n"
        f"B. no\n"
        f"C. maybe\n"
        f"Answer:"
    )


def map_completion_to_answer(raw_completion: str) -> Optional[str]:
    stripped = raw_completion.strip()
    return LETTER_TO_ANSWER.get(stripped)


def score_prediction(gold_answer: str, mapped_prediction: Optional[str]) -> Dict[str, float]:
    return {
        "exact_match": exact_match(gold_answer, mapped_prediction),
        "quasi_exact_match": quasi_exact_match(gold_answer, mapped_prediction),
        "prefix_exact_match": prefix_exact_match(gold_answer, mapped_prediction),
        "quasi_prefix_exact_match": quasi_prefix_exact_match(gold_answer, mapped_prediction),
    }


def aggregate_scores(score_rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(score_rows)
    if not rows:
        return {
            "exact_match": 0.0,
            "quasi_exact_match": 0.0,
            "prefix_exact_match": 0.0,
            "quasi_prefix_exact_match": 0.0,
        }

    metric_names = rows[0].keys()
    return {metric_name: sum(row[metric_name] for row in rows) / len(rows) for metric_name in metric_names}


def load_pubmedqa_items(path: Path) -> List[PubMedQAItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[PubMedQAItem] = []
    for example in data.values():
        gold_answer = example["final_decision"]
        if gold_answer not in POSSIBLE_ANSWER_CHOICES:
            raise ValueError(f"Unexpected gold answer: {gold_answer}")
        items.append(
            PubMedQAItem(
                question=example["QUESTION"],
                labels=example["LABELS"],
                contexts=example["CONTEXTS"],
                gold_answer=gold_answer,
            )
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate HELM PubMedQA prompt construction and metrics.")
    parser.add_argument("--data", type=Path, help="Path to PubMedQA JSON data.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the first prompt and exit.")
    parser.add_argument(
        "--prediction",
        type=str,
        help="Single raw model completion to score for the first example, e.g. A, B, or C.",
    )
    args = parser.parse_args()

    if not args.data:
        parser.error("--data is required")

    items = load_pubmedqa_items(args.data)
    if not items:
        raise ValueError("No examples found in dataset.")

    first_item = items[0]

    if args.show_prompt:
        print(build_helm_prompt(first_item))
        return

    if args.prediction is not None:
        mapped_prediction = map_completion_to_answer(args.prediction)
        result = {
            "raw_prediction": args.prediction,
            "mapped_prediction": mapped_prediction,
            "gold_answer": first_item.gold_answer,
            "scores": score_prediction(first_item.gold_answer, mapped_prediction),
        }
        print(json.dumps(result, indent=2))
        return

    demo_rows = []
    for item in items:
        mapped_prediction = map_completion_to_answer("A")
        demo_rows.append(score_prediction(item.gold_answer, mapped_prediction))

    print(json.dumps(aggregate_scores(demo_rows), indent=2))


if __name__ == "__main__":
    main()
