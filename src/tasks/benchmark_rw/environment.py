import asyncio
import hashlib
import json
import random
import re
import threading
from typing import Any, List, Tuple

from diskcache import Cache
from pydantic import BaseModel, Field

from .prompts import JURY_PROMPT
from .state import StateBenchmarkRW
from ... import EnvironmentFactory
from ...models.online import OnlineLLM
from ...typedefs import Environment, MAX_SEED, Request

cache = Cache(".cache/benchmark_rw_jury_cache")

FINAL_SCORE_THRESHOLD = 3.5
DEFAULT_JURY_SCORE = 1.0


def clean_generation(text: str) -> str:
    text = text.strip()
    for marker in ["Final Answer:", "Final Section:", "Draft:", "Answer:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    text = re.sub(rf"^{re.escape('Candidate draft:')}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(PLAN|SUMMARY|SOLUTION)\s*:\s*", "", text, flags=re.IGNORECASE)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    explanation: str


class JuryEvaluation(BaseModel):
    accuracy: CriterionScore
    completeness: CriterionScore
    clarity: CriterionScore

    def score(self) -> float:
        return (self.accuracy.score + self.completeness.score + self.clarity.score) / 3


@EnvironmentFactory.register
class EnvironmentBenchmarkRW(Environment):
    jury = None
    jury_clients = None

    @staticmethod
    def step(state: StateBenchmarkRW, action: str) -> StateBenchmarkRW:
        cleaned_action = clean_generation(action)
        random.seed(state.randomness if state.randomness is not None else 0)
        randomness = random.randint(0, MAX_SEED)
        return StateBenchmarkRW(
            puzzle=state.puzzle,
            current_state=cleaned_action,
            steps=state.steps + [cleaned_action],
            source_id=state.source_id,
            question=state.question,
            randomness=randomness,
            step_n=state.step_n + 1,
            values=state.values,
        )

    @staticmethod
    def is_valid(state: StateBenchmarkRW, action: str) -> bool:
        return bool(clean_generation(action))

    @staticmethod
    def is_final(state: StateBenchmarkRW) -> bool:
        if not state.steps:
            return False
        return evaluate_with_jury(state) >= FINAL_SCORE_THRESHOLD

    @staticmethod
    def evaluate(state: StateBenchmarkRW) -> Tuple[bool, float]:
        if not state.steps:
            return False, 0.0
        score = evaluate_with_jury(state)
        return score >= FINAL_SCORE_THRESHOLD, float(score)

    @staticmethod
    def add_jury_evaluation(jury_models_info: List[dict]) -> None:
        EnvironmentBenchmarkRW.jury = jury_models_info
        EnvironmentBenchmarkRW.jury_clients = [
            OnlineLLM(
                provider=jury_model_info.get("provider"),
                api_key=jury_model_info.get("api_key"),
                reasoning_effort=jury_model_info.get("reasoning_effort", None),
            )
            for jury_model_info in jury_models_info
        ]


EnvironmentBenchmark_RW = EnvironmentBenchmarkRW


def evaluate_with_jury(state: StateBenchmarkRW) -> float:
    prompt = build_jury_prompt(state.question, state.current_state)
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


def build_jury_prompt(question: str, response: str) -> str:
    return JURY_PROMPT.replace("{QUESTION}", question).replace("{RESPONSE}", response)


def run_jury(prompt: str) -> List[JuryEvaluation]:
    if EnvironmentBenchmarkRW.jury is None or EnvironmentBenchmarkRW.jury_clients is None:
        raise ValueError("Jury models information must be set before running the jury evaluation.")
    if not EnvironmentBenchmarkRW.jury:
        return []

    async def run_single_juror(juror_idx: int) -> JuryEvaluation | None:
        try:
            content = await request_jury_response(prompt, juror_idx, temperature=1.0)
            parsed = parse_jury_evaluation(content)
            if parsed is None:
                raise ValueError("Jury response did not contain a valid evaluation JSON object.")
            return parsed
        except Exception as exc:
            print(f"BenchmarkRW jury error for juror {juror_idx + 1}: {exc}")
        return await run_jury_fallback(prompt, juror_idx)

    async def run_all_jurors() -> List[JuryEvaluation]:
        recovered = await asyncio.gather(
            *(run_single_juror(juror_idx) for juror_idx in range(len(EnvironmentBenchmarkRW.jury)))
        )
        return [evaluation for evaluation in recovered if evaluation is not None]

    return run_async_from_sync(run_all_jurors())


async def run_jury_fallback(prompt: str, juror_idx: int) -> JuryEvaluation | None:
    try:
        content = await request_jury_response(
            prompt=prompt,
            juror_idx=juror_idx,
            temperature=0.0,
            extra_instruction="Return only the valid JSON object requested above.",
        )
        return parse_jury_evaluation(content)
    except Exception as exc:
        print(f"BenchmarkRW fallback jury error for juror {juror_idx + 1}: {exc}")
        return None


async def request_jury_response(
    prompt: str,
    juror_idx: int,
    temperature: float,
    extra_instruction: str | None = None,
) -> str:
    jury_model_info = EnvironmentBenchmarkRW.jury[juror_idx % len(EnvironmentBenchmarkRW.jury)]
    client = EnvironmentBenchmarkRW.jury_clients[juror_idx % len(EnvironmentBenchmarkRW.jury_clients)]
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
            request_id=f"benchmark-rw-jury-{juror_idx}",
            namespace="benchmark_rw_jury",
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
            payload = json.loads(candidate)
            if hasattr(JuryEvaluation, "model_validate"):
                return JuryEvaluation.model_validate(payload)
        except Exception:
            continue
    return None


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


def prompt_cache_key(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
