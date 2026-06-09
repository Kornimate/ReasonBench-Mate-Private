import re
from typing import List

import numpy as np

from . import prompts as prompts
from .environment import ANSWER_TO_LETTER, clean_generation
from .state import StatePubMedQA
from ... import AgentFactory
from ...typedefs import Agent, DecodingParameters, Model
from ...utils import build_population_prediction_prompt, parse_population_prediction


def parse_single_answer(response: str) -> str:
    return clean_generation(response)


def parse_candidate_answers(response: str) -> List[str]:
    matches = re.findall(r"Candidate\s*\d+\s*:\s*(.+?)(?=Candidate\s*\d+\s*:|$)", response, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return [clean_generation(match) for match in matches if clean_generation(match)]

    fallback = [clean_generation(line) for line in response.splitlines()]
    return [line for line in fallback if line]


def build_prompt(template: str, state: StatePubMedQA, current_answer: str = "") -> str:
    answer_letter = ANSWER_TO_LETTER.get(current_answer, "None yet.")
    return template.format(
        prompt_text=state.prompt_text,
        instance_input=state.instance_input,
        current_answer=answer_letter,
        answer=answer_letter,
    )


@AgentFactory.register
class AgentIoPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = prompts.io.format(prompt_text=state.prompt_text)
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_answer(response) for response in responses]


@AgentFactory.register
class AgentCotPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        prompt = build_prompt(prompts.cot, state)
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_answer(response) for response in responses]


@AgentFactory.register
class AgentActPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
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
        return [parse_single_answer(response) for response in responses]


@AgentFactory.register
class AgentPopulationPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
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
class AgentBfsPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
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
        return parse_candidate_answers(responses[0])


@AgentFactory.register
class AgentAggregatePubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        if not actions:
            return []

        action_block = "\n".join(
            f"{idx + 1}. {ANSWER_TO_LETTER.get(action, action)}"
            for idx, action in enumerate(actions)
        )
        prompt = prompts.aggregate.format(
            instance_input=state.instance_input,
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
class AgentReactPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
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
        return [parse_single_answer(response) for response in responses]


async def act_with_pubmedqa_role_prompt(
    template: str,
    model: Model,
    state: StatePubMedQA,
    n: int,
    namespace: str,
    request_id: str,
    params: DecodingParameters,
) -> List[str]:
    prompt = build_prompt(template, state, state.current_state)
    responses = await model.request(
        prompt=prompt,
        n=n,
        request_id=request_id,
        namespace=namespace,
        params=params,
    )
    return [parse_single_answer(response) for response in responses]


@AgentFactory.register
class AgentCriticPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        return await act_with_pubmedqa_role_prompt(
            prompts.critic, model, state, n, namespace, request_id, params
        )


@AgentFactory.register
class AgentCorrectorPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        return await act_with_pubmedqa_role_prompt(
            prompts.corrector, model, state, n, namespace, request_id, params
        )


@AgentFactory.register
class AgentPlannerPubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        return await act_with_pubmedqa_role_prompt(
            prompts.planner, model, state, n, namespace, request_id, params
        )


@AgentFactory.register
class AgentEvaluatePubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        cache_key = (state.source_id, state.current_state)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        prompt = build_prompt(prompts.evaluate, state, state.current_state)
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
class AgentSelfEvaluatePubMedQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StatePubMedQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        cache_key = ("self", state.source_id, state.current_state)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        prompt = build_prompt(prompts.self_evaluate_step, state, state.current_state)
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
