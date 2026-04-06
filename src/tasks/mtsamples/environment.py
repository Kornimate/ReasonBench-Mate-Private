import random
import re
from collections import Counter
from typing import Tuple

from .state import StateMTSamples
from ... import EnvironmentFactory
from ...typedefs import Environment, MAX_SEED


def clean_generation(text: str) -> str:
    text = text.strip()
    if "Final Section:" in text:
        text = text.split("Final Section:")[-1].strip()
    if "Draft:" in text:
        text = text.split("Draft:")[-1].strip()
    text = re.sub(rf"^{re.escape('Candidate draft:')}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(PLAN|SUMMARY|FINDINGS)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_scoring(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_scoring(prediction).split()
    ref_tokens = normalize_for_scoring(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


@EnvironmentFactory.register
class EnvironmentMTSamples(Environment):
    @staticmethod
    def step(state: StateMTSamples, action: str) -> StateMTSamples:
        cleaned_action = clean_generation(action)

        random.seed(state.randomness if state.randomness is not None else 0)
        randomness = random.randint(0, MAX_SEED)

        return StateMTSamples(
            puzzle=state.puzzle,
            current_state=cleaned_action,
            steps=state.steps + [cleaned_action],
            answer=state.answer,
            section_name=state.section_name,
            title=state.title,
            source_id=state.source_id,
            note_text=state.note_text,
            randomness=randomness,
            step_n=state.step_n + 1,
            values=state.values,
        )

    @staticmethod
    def is_valid(state: StateMTSamples, action: str) -> bool:
        return bool(clean_generation(action))

    @staticmethod
    def is_final(state: StateMTSamples) -> bool:
        if not state.steps:
            return False
        score = token_f1(state.current_state, state.answer)
        return score >= 0.8

    @staticmethod
    def evaluate(state: StateMTSamples) -> Tuple[bool, float]:
        if not state.steps:
            return False, 0.0
        score = token_f1(state.current_state, state.answer)
        return True, float(score)
