import re
from typing import List

from . import prompts
from .environment import answers_match, clean_answer, extract_final_answer
from .state import StateMathArena
from ... import AgentFactory
from ...typedefs import Agent, DecodingParameters, Model
from ...utils import build_population_prediction_prompt, parse_population_prediction


ACTION_PATTERN = re.compile(r"(Analyze|Explain|Finish)\[(.*?)\]", re.IGNORECASE | re.DOTALL)
GENERIC_ACTION_CONTENT = {
    "analysis",
    "concept",
    "concepts",
    "math concepts",
    "problem",
    "problem statement",
    "solution approach",
    "solution steps",
    "underlying problem structure",
    "underlying structure",
}


@AgentFactory.register
class AgentIoMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        responses = await model.request(
            prompt=prompts.io.format(input=state.problem),
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [to_finish_action(response) for response in responses]


@AgentFactory.register
class AgentCotMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        responses = await model.request(
            prompt=prompts.cot.format(input=state.problem),
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [to_finish_action(response) for response in responses]


@AgentFactory.register
class AgentActMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        responses = await model.request(
            prompt=prompts.act.format(input=state.problem),
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_action(response) for response in responses]


@AgentFactory.register
class AgentReactMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        responses = await model.request(
            prompt=prompts.react.format(input=state.problem),
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return [parse_single_action(response) for response in responses]


@AgentFactory.register
class AgentPopulationMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        max_agents: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> int:
        response = await model.request(
            prompt=build_population_prediction_prompt(state, max_agents),
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return parse_population_prediction(response[0], max_agents, max_agents)


@AgentFactory.register
class AgentBfsMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        responses = await model.request(
            prompt=prompts.bfs.format(input=state.problem),
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return parse_actions(responses[0])


@AgentFactory.register
class AgentAggregateMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        if not actions:
            return []
        if len(actions) <= k:
            return actions

        action_block = "\n".join(f"({i + 1}) {action}" for i, action in enumerate(actions))
        responses = await model.request(
            prompt=prompts.aggregate.format(
                input=state.problem,
                actions=action_block,
                k=k,
            ),
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        indexes = [int(index) - 1 for index in re.findall(r"\d+", responses[0])]
        selected = [actions[index] for index in indexes if 0 <= index < len(actions)]
        return selected[:k] if selected else actions[:k]


@AgentFactory.register
class AgentEvaluateMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        cache_key = state.current_state
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        is_final, reward = local_score(state)
        if is_final:
            value = 20.0 if reward == 1.0 else 0.001
        else:
            responses = await model.request(
                prompt=prompts.evaluate.format(
                    input=state.problem,
                    steps=format_steps(state),
                ),
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )
            values = [parse_value(response) for response in responses]
            value = sum(values) / len(values) if values else 0.0

        state.values[state.step_n] = value
        if cache is not None:
            cache[cache_key] = value
        return value


@AgentFactory.register
class AgentSelfEvaluateMathArena(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateMathArena,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        cache_key = f"self:{state.current_state}"
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        is_final, reward = local_score(state)
        if is_final:
            value = 20.0 if reward == 1.0 else 0.001
        else:
            latest_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                input=state.problem,
                previous_steps="\n".join(state.steps[:-1]) or "None",
                step=latest_step,
            )
            responses = await model.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )
            votes = [1.0 if response.strip().lower().startswith("yes") else 0.0 for response in responses]
            value = sum(votes) / len(votes) if votes else 0.001

        if cache is not None:
            cache[cache_key] = value
        return value


def format_steps(state: StateMathArena) -> str:
    return "\n".join(state.steps) if state.steps else "None"


def parse_actions(response: str) -> List[str]:
    actions = parse_actions_no_fallback(response)
    if actions:
        return actions

    lines = [line.strip(" -\t") for line in str(response).splitlines() if line.strip()]
    parsed = [parse_single_action(line) for line in lines]
    return parsed or [parse_single_action(response)]


def parse_single_action(response: str) -> str:
    response = str(response)
    actions = parse_actions_no_fallback(response)
    if actions:
        finish_actions = [action for action in actions if action.startswith("Finish[")]
        if finish_actions:
            return finish_actions[-1]
        return actions[0]

    final = extract_final_answer(response)
    if final is not None:
        return f"Finish[{final}]"

    text = clean_answer(response)
    if not text:
        text = "No useful step was produced."
    return f"Explain[{text}]"


def parse_actions_no_fallback(response: str) -> List[str]:
    response = str(response)
    matches = list(ACTION_PATTERN.finditer(response))
    actions = []
    for index, match in enumerate(matches):
        kind = match.group(1)
        content = clean_answer(match.group(2))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        trailing = clean_answer(response[match.end():end])
        if content.lower() in GENERIC_ACTION_CONTENT and trailing:
            content = trailing
        elif trailing:
            content = f"{content} {trailing}"
        actions.append(format_action(kind, content))
    return actions


def format_action(kind: str, content: str) -> str:
    kind = kind.capitalize()
    content = clean_answer(content)
    return f"{kind}[{content}]"


def to_finish_action(response: str) -> str:
    final = extract_final_answer(response)
    if final is None:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", str(response))
        final = boxed[-1] if boxed else str(response).strip().splitlines()[-1]
    return f"Finish[{clean_answer(final)}]"


def parse_value(response: str, low: float = 0.0, high: float = 1.0) -> float:
    response = str(response).lower()
    if "score" in response:
        response = response.split("score", 1)[-1]
    numbers = re.findall(r"-?[0-9]+\.?[0-9]*", response)
    if numbers:
        value = float(numbers[-1])
        return min(max(low, value), high)
    if "solvable" in response:
        return high
    if "likely" in response:
        return 0.5
    if "needs more info" in response or "impossible" in response:
        return low
    if "incorrect" in response or "wrong" in response:
        return low
    if "correct" in response or "promising" in response:
        return high
    return low


def local_score(state: StateMathArena) -> tuple[bool, float]:
    if not state.steps:
        return False, 0.0
    final = extract_final_answer(state.steps[-1])
    if final is None:
        return False, 0.0
    return True, float(answers_match(final, state.answer))
