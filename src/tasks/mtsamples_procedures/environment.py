import asyncio
import hashlib
import json
import random
import re
import threading
from typing import Any, List, Tuple

from diskcache import Cache
from pydantic import BaseModel, Field

from .prompts import JURY_PROMPT, io as TASK_PROMPT
from .state import StateMTSamplesProcedures
from ... import EnvironmentFactory
from ...models.online import OnlineLLM
from ...typedefs import Environment, MAX_SEED, Request

cache = Cache(".cache/mtsamples_procedures_jury_cache")

MAX_LIKERT_SCORE = 5.0 # based on MTSamplesProcedures evaluation criteria
FINAL_SCORE_THRESHOLD = 3.8
DEFAULT_JURY_SCORE = 1.0

# parsing llm response for jury evaluation, introduced robostusess with possible variations of response formatting
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

    def score(self) -> float:
        raw_scores = [
            self.accuracy.score,
            self.completeness.score,
            self.clarity.score,
        ]
        return sum(raw_scores) / len(raw_scores)

@EnvironmentFactory.register
class EnvironmentMTSamplesProcedures(Environment):
    
    jury = None # holder for llm-as-a-jury information
    jury_clients = None
    
    @staticmethod
    def step(state: StateMTSamplesProcedures, action: str) -> StateMTSamplesProcedures:
        cleaned_action = clean_generation(action)

        random.seed(state.randomness if state.randomness is not None else 0)
        randomness = random.randint(0, MAX_SEED)

        return StateMTSamplesProcedures(
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
    def is_valid(state: StateMTSamplesProcedures, action: str) -> bool:
        return bool(clean_generation(action))

    @staticmethod
    def is_final(state: StateMTSamplesProcedures) -> bool:
        if not state.steps:
            return False
        score = evaluate_with_jury(state)
        return score >= FINAL_SCORE_THRESHOLD

    @staticmethod
    def evaluate(state: StateMTSamplesProcedures) -> Tuple[bool, float]:
        if not state.steps:
            return False, 0.0
        score = evaluate_with_jury(state)
        return score >= FINAL_SCORE_THRESHOLD, float(score)
    
    @staticmethod
    def add_jury_evaluation(jury_models_info: List[dict]) -> None:
        EnvironmentMTSamplesProcedures.jury = jury_models_info
        EnvironmentMTSamplesProcedures.jury_clients = [
            OnlineLLM(
                provider=jury_model_info.get("provider"),
                api_key=jury_model_info.get("api_key"),
                reasoning_effort=jury_model_info.get("reasoning_effort", None),
            )
            for jury_model_info in jury_models_info
        ]

def evaluate_with_jury(state: StateMTSamplesProcedures) -> float:
    question = build_user_request(state)
    prompt = build_jury_prompt(question, state.current_state, state.answer)
    key = prompt_cache_key(prompt)
    cached_score = cache.get(key)
    if cached_score is not None:
        return float(cached_score)

    evaluations = run_jury(prompt)
    if not evaluations:
        return DEFAULT_JURY_SCORE

    score = sum(evaluation.score() for evaluation in evaluations) / len(evaluations)
    cache.set(key, score)
    return float(score)


def build_jury_prompt(question: str, response: str, gold_response: str) -> str:
    return (
        JURY_PROMPT
        .replace("{QUESTION}", question)
        .replace("{RESPONSE}", response)
        .replace("{GOLD_RESPONSE}", gold_response)
    )


def run_jury(prompt: str) -> List[JuryEvaluation]:
    if EnvironmentMTSamplesProcedures.jury is None or EnvironmentMTSamplesProcedures.jury_clients is None:
        raise ValueError("Jury models information must be set before running the jury evaluation.")
    if not EnvironmentMTSamplesProcedures.jury:
        return []

    async def run_single_juror(juror_idx: int) -> JuryEvaluation | None:
        try:
            content = await request_jury_response(prompt, juror_idx, temperature=1.0)
            parsed = parse_jury_evaluation(content)
            if parsed is None:
                raise ValueError("Jury response did not contain a valid evaluation JSON object.")
            return parsed
        except Exception as exc:
            print(f"MTSamplesProcedures jury error for juror {juror_idx + 1}: {exc}")

        return await run_jury_fallback(prompt, juror_idx)

    async def run_all_jurors() -> List[JuryEvaluation]:
        recovered = await asyncio.gather(
            *(run_single_juror(juror_idx) for juror_idx in range(len(EnvironmentMTSamplesProcedures.jury)))
        )
        return [evaluation for evaluation in recovered if evaluation is not None]

    return run_async_from_sync(run_all_jurors())


def build_user_request(state: StateMTSamplesProcedures) -> str:
    cleaned_text = f"Procedure note title: {state.title}\n\nNote:\n{state.note_text}"
    return TASK_PROMPT.format(cleaned_text=cleaned_text)


async def run_jury_fallback(prompt: str, juror_idx: int) -> JuryEvaluation | None:
    try:
        content = await request_jury_response(
            prompt=prompt,
            juror_idx=juror_idx,
            temperature=0.0,
            extra_instruction=(
                "Return only the valid JSON object requested above. "
                "Do not include markdown fences or any extra text."
            ),
        )
        return parse_jury_evaluation(content)
    except Exception as exc:
        print(f"MTSamplesProcedures fallback jury error for juror {juror_idx + 1}: {exc}")
        return None


async def request_jury_response(
    prompt: str,
    juror_idx: int,
    temperature: float,
    extra_instruction: str | None = None,
) -> str:
    jury_model_info = EnvironmentMTSamplesProcedures.jury[juror_idx % len(EnvironmentMTSamplesProcedures.jury)]
    client = EnvironmentMTSamplesProcedures.jury_clients[juror_idx % len(EnvironmentMTSamplesProcedures.jury_clients)]
    messages: str | List[dict[str, str]]
    messages = [{"role": "user", "content": prompt}]
    if extra_instruction:
        messages.append({"role": "user", "content": extra_instruction})

    response = await client.request(
        Request(
            args="",
            kwargs={
                "prompt": messages,
                "model": jury_model_info["model"],
                "max_completion_tokens": 1024,
                "temperature": temperature,
                "top_p": jury_model_info.get("top_p", 1.0),
                "stop": jury_model_info.get("stop"),
                "logprobs": False,
            },
            n=1,
            request_id=f"mtsamples-procedures-jury-{juror_idx}",
            namespace="mtsamples_procedures_jury",
        )
    )
    if not response.data:
        return ""
    first_result = response.data[0]
    if isinstance(first_result, tuple):
        return str(first_result[0] or "")
    return str(first_result or "")


def run_async_from_sync(coroutine) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def run_in_thread() -> None:
        try:
            result["value"] = asyncio.run(coroutine)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    if error:
        raise error["value"]
    return result.get("value")


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
