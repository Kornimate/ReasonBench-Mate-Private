import asyncio
import random
from typing import TypedDict

from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, MAX_SEED, Method, Model, State
from ..utils import Resampler


class StepAgentSpec(TypedDict):
    agent: Agent
    params: DecodingParameters
    num_agents: int


@AgentDictFactory.register
class AgentDictHetFOA(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]


@MethodFactory.register
class MethodHetFOA(Method):
    def __init__(
        self,
        agents: AgentDictHetFOA,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(model, agents, env, config)

        self.eval_agent = agents["evaluate"]
        self.evaluate_params = agents["evaluate_params"]
        self.step_agents = agents["step_agents"]

        self.num_agents = int(config.num_agents)
        self.num_steps = int(config.num_steps)
        self.k = int(config.k)
        self.backtrack = float(config.backtrack)
        self.resampling = str(config.resampling)
        self.origin = float(config.origin)
        self.min_steps = int(config.min_steps)
        self.num_evaluations = int(config.num_evaluations)

        total_agents = sum(spec["num_agents"] for spec in self.step_agents)
        if total_agents != self.num_agents:
            raise ValueError(
                f"het_foa expected {self.num_agents} total step agents, got {total_agents}"
            )

    def _get_agent_spec(self, index: int) -> StepAgentSpec:
        remaining = index
        for spec in self.step_agents:
            if remaining < spec["num_agents"]:
                return spec
            remaining -= spec["num_agents"]
        raise IndexError(f"Step agent index {index} out of range")

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        random.seed(idx)
        resampler = Resampler(idx)

        visited_states: list[tuple[str, float, State]] = [("INIT", self.origin, state)]
        initial_state = state
        states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.num_agents)]

        for step in range(self.num_steps):
            action_coroutines = []
            for i, candidate_state in enumerate(states):
                spec = self._get_agent_spec(i)
                action_coroutines.append(
                    spec["agent"].act(
                        model=self.model,
                        state=candidate_state,
                        n=1,
                        namespace=namespace,
                        request_id=f"idx{idx}-step{step}-{hash(candidate_state)}-agent{i}",
                        params=spec["params"],
                    )
                )

            actions_per_state = await asyncio.gather(*action_coroutines)

            next_states = []
            for candidate_state, actions in zip(states, actions_per_state):
                if not actions:
                    next_states.append(candidate_state)
                    continue
                try:
                    next_states.append(self.env.step(candidate_state, actions[0]))
                except Exception:
                    next_states.append(candidate_state)
            states = next_states

            if any(self.env.evaluate(candidate_state)[1] == 1 for candidate_state in states):
                return states

            remaining_steps = self.num_steps - (step + 1)
            visited_states = [
                (identifier, value * self.backtrack, candidate_state)
                for identifier, value, candidate_state in visited_states
                if remaining_steps >= self.min_steps - len(candidate_state.steps)
            ]

            finished = [i for i, candidate_state in enumerate(states) if self.env.is_final(candidate_state)]
            if finished:
                if visited_states:
                    replacements, _ = resampler.resample(visited_states.copy(), len(finished), self.resampling)
                else:
                    replacements, _ = resampler.resample(
                        [("", 1.0, candidate_state) for candidate_state in states],
                        len(finished),
                        "linear",
                    )
                states = [replacements.pop(0) if i in finished else candidate_state for i, candidate_state in enumerate(states)]

            if step < self.num_steps - 1 and self.k and step % self.k == 0:
                value_coroutines = [
                    self.eval_agent.act(
                        model=self.model,
                        state=candidate_state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{step}-{hash(candidate_state)}-agent{i}",
                        params=self.evaluate_params,
                        cache=value_cache,
                    )
                    for i, candidate_state in enumerate(states)
                ]
                values = await asyncio.gather(*value_coroutines)

                for i, (candidate_state, value) in enumerate(zip(states, values)):
                    if i not in finished:
                        visited_states.append((f"{i}.{step}", float(value), candidate_state))

                states, _ = resampler.resample(visited_states, self.num_agents, self.resampling)

        return states if states else [initial_state]
