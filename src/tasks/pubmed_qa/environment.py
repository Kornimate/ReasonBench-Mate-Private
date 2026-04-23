import random
import re
import string
from typing import Dict, Optional, Tuple

from .state import StatePubMedQA
from ... import EnvironmentFactory
from ...typedefs import Environment, MAX_SEED

POSSIBLE_ANSWER_CHOICES = ["yes", "no", "maybe"]
LETTER_TO_ANSWER = {"A": "yes", "B": "no", "C": "maybe"}
ANSWER_TO_LETTER = {value: key for key, value in LETTER_TO_ANSWER.items()}


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


def score_prediction(gold_answer: str, mapped_prediction: Optional[str]) -> Dict[str, float]:
    return {
        "exact_match": exact_match(gold_answer, mapped_prediction),
        "quasi_exact_match": quasi_exact_match(gold_answer, mapped_prediction),
        "prefix_exact_match": prefix_exact_match(gold_answer, mapped_prediction),
        "quasi_prefix_exact_match": quasi_prefix_exact_match(gold_answer, mapped_prediction),
    }


def _extract_candidate(text: str) -> str:
    cleaned = text.strip()
    for marker in ["Final Answer:", "Answer:", "Draft:", "Candidate 1:", "Candidate:"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
    return cleaned


def map_completion_to_answer(raw_completion: str) -> Optional[str]:
    stripped = _extract_candidate(raw_completion)
    if not stripped:
        return None

    first_token_match = re.search(r"\b([ABC])\b", stripped, flags=re.IGNORECASE)
    if first_token_match:
        return LETTER_TO_ANSWER.get(first_token_match.group(1).upper())

    normalized = normalize_text(stripped, should_remove_articles=False)
    for answer in POSSIBLE_ANSWER_CHOICES:
        if normalized == answer or normalized.startswith(f"{answer} "):
            return answer

    return None


def clean_generation(text: str) -> str:
    return map_completion_to_answer(text) or ""


@EnvironmentFactory.register
class EnvironmentPubMedQA(Environment):
    @staticmethod
    def step(state: StatePubMedQA, action: str) -> StatePubMedQA:
        cleaned_action = clean_generation(action)

        random.seed(state.randomness if state.randomness is not None else 0)
        randomness = random.randint(0, MAX_SEED)

        return StatePubMedQA(
            prompt_text=state.prompt_text,
            instance_input=state.instance_input,
            question=state.question,
            current_state=cleaned_action,
            steps=state.steps + [cleaned_action],
            answer=state.answer,
            source_id=state.source_id,
            randomness=randomness,
            step_n=state.step_n + 1,
            values=state.values,
            metrics=state.metrics,
        )

    @staticmethod
    def is_valid(state: StatePubMedQA, action: str) -> bool:
        return bool(clean_generation(action))

    @staticmethod
    def is_final(state: StatePubMedQA) -> bool:
        if not state.steps:
            return False
        scores = evaluate_prediction(state)
        return bool(scores["exact_match"] == 1.0)

    @staticmethod
    def evaluate(state: StatePubMedQA) -> Tuple[bool, float]:
        if not state.steps:
            state.metrics["evaluation"] = build_named_metrics({})
            return False, 0.0

        scores = evaluate_prediction(state)
        named_metrics = build_named_metrics(scores)
        state.metrics["evaluation"] = named_metrics
        return bool(scores["exact_match"] == 1.0), float(scores["exact_match"])


def build_named_metrics(scores: Dict[str, float]) -> Dict[str, Dict[str, float | str]]:
    labels = {
        "exact_match": "Exact Match",
        "quasi_exact_match": "Quasi Exact Match",
        "prefix_exact_match": "Prefix Exact Match",
        "quasi_prefix_exact_match": "Quasi Prefix Exact Match",
    }
    return {
        key: {
            "label": labels[key],
            "score": float(scores.get(key, 0.0)),
        }
        for key in labels
    }


def evaluate_prediction(state: StatePubMedQA) -> Dict[str, float]:
    return score_prediction(state.answer, state.current_state)
