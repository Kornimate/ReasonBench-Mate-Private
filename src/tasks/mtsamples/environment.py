import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, override

from diskcache import Cache
from openai import OpenAI
from pydantic import BaseModel, Field

from .prompts import JURY_PROMPT, io as TASK_PROMPT
from .state import StateMTSamples
from ... import EnvironmentFactory
from ...typedefs import Environment, MAX_SEED

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_CLAN") or os.getenv("OPENAI_API_KEY"), timeout=300, max_retries=1)
cache = Cache(".cache/mtsamples_jury_cache")

JUDGE_MODEL = os.getenv("MTSAMPLES_JUDGE_MODEL", "gpt-4.1-nano")
JURY_SIZE = int(os.getenv("MTSAMPLES_JURY_SIZE", "3")) # if specified other way but standard is 3 based on MedHelm description
MAX_LIKERT_SCORE = 5.0 # based on MTSamples evaluation criteria


def clean_generation(text: str) -> str:
    text = text.strip()
    if "Final Section:" in text:
        text = text.split("Final Section:")[-1].strip()
    if "Draft:" in text:
        text = text.split("Draft:")[-1].strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1].strip()
    text = re.sub(rf"^{re.escape('Candidate draft:')}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(PLAN|SUMMARY|FINDINGS)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    explanation: str


class JuryEvaluation(BaseModel):
    accuracy: CriterionScore
    completeness: CriterionScore
    clarity: CriterionScore

    def normalized_score(self) -> float:
        raw_scores = [
            self.accuracy.score,
            self.completeness.score,
            self.clarity.score,
        ]
        return sum(raw_scores) / (len(raw_scores) * MAX_LIKERT_SCORE)


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
        score = evaluate_with_jury(state)
        return score >= 0.9

    @staticmethod
    def evaluate(state: StateMTSamples) -> Tuple[bool, float]:
        if not state.steps:
            return False, 0.0
        score = evaluate_with_jury(state)
        return True, float(score)
    
    @staticmethod
    @override
    def add_jury_evaluation(jury_models_info: List[dict]) -> None:
        pass

def evaluate_with_jury(state: StateMTSamples) -> float:
    question = build_user_request(state)
    prompt = JURY_PROMPT.format(
        QUESTION=question,
        RESPONSE=state.current_state,
        GOLD_RESPONSE=state.answer,
    )
    key = prompt_cache_key(prompt)
    cached_score = cache.get(key)
    if cached_score is not None:
        return float(cached_score)

    evaluations = run_jury(prompt)
    if not evaluations:
        return 0.0

    score = sum(evaluation.normalized_score() for evaluation in evaluations) / len(evaluations)
    cache.set(key, score)
    return float(score)


def run_jury(prompt: str) -> List[JuryEvaluation]:
    if JURY_SIZE <= 0:
        return []

    def run_single_juror(juror_idx: int) -> JuryEvaluation | None:
        try:
            response = client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                max_completion_tokens=1024,
                temperature=1.0,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format=JuryEvaluation,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception as exc:
            print(f"MTSamples structured jury error for juror {juror_idx + 1}: {exc}")

        return run_jury_fallback(prompt, juror_idx)

    evaluations: List[JuryEvaluation] = []
    with ThreadPoolExecutor(max_workers=JURY_SIZE) as executor:
        futures = [executor.submit(run_single_juror, juror_idx) for juror_idx in range(JURY_SIZE)]
        for future in as_completed(futures):
            recovered = future.result()
            if recovered is not None:
                evaluations.append(recovered)
    return evaluations


def build_user_request(state: StateMTSamples) -> str:
    cleaned_text = f"Procedure note title: {state.title}\n\nNote:\n{state.note_text}"
    return TASK_PROMPT.format(cleaned_text=cleaned_text)


def run_jury_fallback(prompt: str, juror_idx: int) -> JuryEvaluation | None:
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            max_completion_tokens=1024,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": "Return only the valid JSON object requested above. Do not include markdown fences or any extra text.",
                },
            ],
        )
        content = response.choices[0].message.content or ""
        return parse_jury_evaluation(content)
    except Exception as exc:
        print(f"MTSamples fallback jury error for juror {juror_idx + 1}: {exc}")
        return None


def parse_jury_evaluation(content: str) -> JuryEvaluation | None:
    for candidate in json_candidates(content):
        try:
            return validate_jury_payload(json.loads(candidate))
        except Exception:
            continue
    return None


def validate_jury_payload(payload: dict) -> JuryEvaluation:
    if hasattr(JuryEvaluation, "model_validate"):
        return JuryEvaluation.model_validate(payload)
    raise ValueError("Error while validating jury payload: Pydantic v2 is required for model validation.")


def json_candidates(content: str) -> List[str]:
    cleaned = content.strip()
    candidates = [cleaned]

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0).strip())

    return candidates


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def prompt_cache_key(prompt: str) -> str:
    normalized = normalize_prompt(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
