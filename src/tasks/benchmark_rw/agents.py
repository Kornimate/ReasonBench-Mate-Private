import re
from typing import List

import numpy as np

from . import prompts
from .environment import clean_generation
from .state import StateBenchmarkRW
from ... import AgentFactory
from ...typedefs import Agent, DecodingParameters, Model
from ...utils import build_population_prediction_prompt, parse_population_prediction


def parse_single_draft(response: str) -> str:
    if "Final Answer:" in response:
        response = response.split("Final Answer:")[-1]
    if "Draft:" in response:
        response = response.split("Draft:")[-1]
    return clean_generation(response)


def parse_candidate_drafts(response: str) -> List[str]:
    matches = re.findall(r"Candidate\s*\d+\s*:\s*(.+?)(?=Candidate\s*\d+\s*:|$)", response, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return [clean_generation(match) for match in matches if clean_generation(match)]
    fallback = [clean_generation(line) for line in response.splitlines()]
    return [line for line in fallback if line]


def build_prompt(template: str, state: StateBenchmarkRW, current_draft: str = "None yet.") -> str:
    return template.format(question=state.question, current_draft=current_draft or "None yet.")


@AgentFactory.register
class AgentIoBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        responses = await model.request(prompt=prompts.io.format(question=state.question), n=n, request_id=request_id, namespace=namespace, params=params)
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentCotBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        responses = await model.request(prompt=prompts.cot.format(question=state.question), n=n, request_id=request_id, namespace=namespace, params=params)
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentActBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        responses = await model.request(prompt=build_prompt(prompts.act, state, state.current_state), n=n, request_id=request_id, namespace=namespace, params=params)
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentReactBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        responses = await model.request(prompt=build_prompt(prompts.react, state, state.current_state), n=n, request_id=request_id, namespace=namespace, params=params)
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentBfsBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        responses = await model.request(prompt=build_prompt(prompts.bfs, state, state.current_state), n=1, request_id=request_id, namespace=namespace, params=params)
        return parse_candidate_drafts(responses[0])


@AgentFactory.register
class AgentAggregateBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        if not actions:
            return []
        action_block = "\n".join(f"{idx + 1}. {action}" for idx, action in enumerate(actions))
        prompt = prompts.aggregate.format(question=state.question, actions=action_block, k=k)
        responses = await model.request(prompt=prompt, n=1, request_id=request_id, namespace=namespace, params=params)
        indexes = [int(value) - 1 for value in re.findall(r"\d+", responses[0])]
        return [actions[index] for index in indexes if 0 <= index < len(actions)]


@AgentFactory.register
class AgentEvaluateBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict = None) -> float:
        cache_key = (state.source_id, state.current_state)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        prompt = prompts.evaluate.format(question=state.question, current_draft=state.current_state or "No solution yet.")
        responses = await model.request(prompt=prompt, n=n, request_id=request_id, namespace=namespace, params=params)
        values = [parse_value(response) for response in responses]
        value = sum(values) / len(values) if values else 0.0
        if cache is not None:
            cache[cache_key] = value
        state.values[state.step_n] = value
        return value


@AgentFactory.register
class AgentSelfEvaluateBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict = None) -> float:
        cache_key = ("self", state.source_id, state.current_state)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        evaluate_params = DecodingParameters(params.max_completion_tokens, params.temperature, params.top_p, params.stop, True)
        prompt = prompts.self_evaluate_step.format(question=state.question, draft=state.current_state or "No solution yet.")
        responses = await model.request(prompt=prompt, n=n, request_id=request_id, namespace=namespace, params=evaluate_params)
        yes_probabilities = []
        for response in responses:
            if hasattr(response, "logprobs") and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                yes_prob = next((prob for token, prob in first_token_logprobs.items() if token.lower() in ["yes", "yes.", "yes!"]), 0.0)
                yes_probabilities.append(np.exp(yes_prob))
        value = (sum(yes_probabilities) / len(yes_probabilities)) if yes_probabilities else 0.001
        if cache is not None:
            cache[cache_key] = value
        return value


@AgentFactory.register
class AgentPopulationBenchmarkRW(Agent):
    @staticmethod
    async def act(model: Model, state: StateBenchmarkRW, max_agents: int, namespace: str, request_id: str, params: DecodingParameters) -> int:
        response = await model.request(
            prompt=build_population_prediction_prompt(state, max_agents),
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return parse_population_prediction(response[0], max_agents, max_agents)


def parse_value(response: str, low: float = 0.0, high: float = 1.0) -> float:
    cleaned = response.strip()
    if "score" not in cleaned.lower():
        return low
    try:
        match = re.search(
            r"score\s*(?:is|=|:)?\s*\**\s*(-?[0-9]+(?:\.[0-9]+)?)\s*(?:/\s*(-?[0-9]+(?:\.[0-9]+)?))?",
            cleaned,
            flags=re.IGNORECASE,
        )
        if match is None:
            match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:/\s*(-?[0-9]+(?:\.[0-9]+)?))?\s*$", cleaned)
        if match is None:
            return low
        value = float(match.group(1))
        denominator = float(match.group(2)) if match.lastindex and match.group(2) else None
        if denominator and denominator > 0:
            value = value / denominator
        elif value > high and value <= 10:
            value = value / 10
        return min(max(low, value), high)
    except Exception:
        return low
