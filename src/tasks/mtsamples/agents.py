import re
from typing import List

import numpy as np

from . import prompts as prompts
from .environment import clean_generation
from .state import StateMTSamples
from ... import AgentFactory
from ...typedefs import Agent, DecodingParameters, Model
from ...utils import build_population_prediction_prompt, parse_population_prediction


def parse_single_draft(response: str) -> str:
    if "Final Section:" in response:
        response = response.split("Final Section:")[-1]
    if "Draft:" in response:
        response = response.split("Draft:")[-1]
    return clean_generation(response)


def parse_candidate_drafts(response: str) -> List[str]:
    matches = re.findall(r"Candidate\s*\d+\s*:\s*(.+?)(?=Candidate\s*\d+\s*:|$)", response, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return [clean_generation(match) for match in matches if clean_generation(match)]

    fallback = [clean_generation(line) for line in response.splitlines()]
    return [line for line in fallback if line]


def build_cleaned_text(state: StateMTSamples) -> str:
    return f"Procedure note title: {state.title}\n\nNote:\n{state.note_text}"


def build_prompt(template: str, state: StateMTSamples, current_draft: str = "None yet.") -> str:
    return template.format(
        cleaned_text=build_cleaned_text(state),
        current_draft=current_draft or "None yet.",
    )


@AgentFactory.register
class AgentIoMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = prompts.io.format(
            cleaned_text=build_cleaned_text(state),
        )
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentCotMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = prompts.cot.format(
            cleaned_text=build_cleaned_text(state),
        )
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentActMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = build_prompt(prompts.act, state, state.current_state)
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentPopulationMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        max_agents: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> int:
        prompt = build_population_prediction_prompt(state, max_agents)
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return parse_population_prediction(response[0], max_agents, max_agents)


@AgentFactory.register
class AgentBfsMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = build_prompt(prompts.bfs, state, state.current_state)
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return parse_candidate_drafts(responses[0])


@AgentFactory.register
class AgentAggregateMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        if not actions:
            return []

        action_block = "\n".join(
            f"{idx + 1}. {action}" for idx, action in enumerate(actions)
        )
        prompt = prompts.aggregate.format(
            cleaned_text=build_cleaned_text(state),
            actions=action_block,
            k=k,
        )
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        try:
            indexes = [int(value) - 1 for value in re.findall(r"\d+", responses[0])]
            return [actions[index] for index in indexes if 0 <= index < len(actions)]
        except Exception:
            return []


@AgentFactory.register
class AgentReactMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = build_prompt(prompts.react, state, state.current_state)
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_draft(response) for response in responses]


@AgentFactory.register
class AgentEvaluateMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        cache_key = (state.title, state.current_state)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        prompt = prompts.evaluate.format(
            cleaned_text=build_cleaned_text(state),
            current_draft=state.current_state or "No draft yet.",
        )
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        values = [parse_value(response) for response in responses]
        value = sum(values) / len(values) if values else 0.0

        if cache is not None:
            cache[cache_key] = value
        state.values[state.step_n] = value
        return value


@AgentFactory.register
class AgentSelfEvaluateMTSamples(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMTSamples,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        cache_key = ("self", state.title, state.current_state)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        prompt = prompts.self_evaluate_step.format(
            note_text=state.note_text,
            draft=state.current_state or "No draft yet.",
        )
        evaluate_params = DecodingParameters(
            temperature=params.temperature,
            max_completion_tokens=params.max_completion_tokens,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=True,
        )
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=evaluate_params,
        )

        yes_probabilities = []
        for response in responses:
            if hasattr(response, "logprobs") and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                yes_prob = next(
                    (
                        prob
                        for token, prob in first_token_logprobs.items()
                        if token.lower() in ["yes", "yes.", "yes!"]
                    ),
                    0.0,
                )
                yes_probabilities.append(np.exp(yes_prob))

        value = (sum(yes_probabilities) / len(yes_probabilities)) if yes_probabilities else 0.001
        if cache is not None:
            cache[cache_key] = value
        return value


def parse_value(response: str, low: float = 0.0, high: float = 1.0) -> float:
    if "score" not in response.lower():
        return low
    try:
        match = re.search(r"score\s*:\s*(-?[0-9]+\.?[0-9]*)", response, flags=re.IGNORECASE)
        if match is None:
            match = re.search(r"(-?[0-9]+\.?[0-9]*)\s*$", response.strip())
        if match is None:
            return low
        value = float(match.group(1))
        return min(max(low, value), high)
    except Exception:
        return low
