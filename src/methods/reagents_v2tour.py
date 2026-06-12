import random
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from ..logging_utils import log_event
from .reagents import DifficultyAgentSpec, MethodReagents, SearchRecord, StepAgentSpec


@AgentDictFactory.register
class AgentDictReagents_v2tour(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v2tour(MethodReagents):
    def __init__(
        self,
        agents: AgentDictReagents_v2tour,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.tournament_size = int(getattr(config, "tournament_size", 2))
        self.tournament_epsilon = float(getattr(config, "tournament_epsilon", 0.0))

    def _clone_selected_states(self, resampler, visited_states, selected_indices):
        random.seed(resampler.randomness)
        new_randomness = [random.randint(1, 1000) for _ in selected_indices]
        if new_randomness:
            resampler.randomness = new_randomness[-1]
        return [
            visited_states[source_index][2].clone(randomness)
            for source_index, randomness in zip(selected_indices, new_randomness)
        ]

    def _tournament_state_indices(self, visited_states, width: int) -> list[int]:
        if not visited_states or width <= 0:
            return []

        tournament_size = max(1, min(self.tournament_size, len(visited_states)))
        epsilon = max(0.0, min(1.0, self.tournament_epsilon))
        selected_indices = []
        tournament_logs = []

        for _ in range(width):
            if np.random.random() < epsilon:
                candidates = [int(np.random.choice(len(visited_states)))]
                selected = candidates[0]
            else:
                candidates = np.random.choice(
                    len(visited_states),
                    size=tournament_size,
                    replace=False,
                ).tolist()
                selected = max(candidates, key=lambda source_index: visited_states[source_index][1])

            selected_indices.append(int(selected))
            tournament_logs.append({
                "candidate_indices": candidates,
                "candidate_values": [float(visited_states[i][1]) for i in candidates],
                "selected_index": int(selected),
                "selected_value": float(visited_states[selected][1]),
            })

        log_event("REAGENTS_V2TOUR_STATE_SELECTION", {
            "width": width,
            "tournament_size": tournament_size,
            "selected_indices": selected_indices,
            "tournaments": tournament_logs,
        })
        return selected_indices

    def _resample_records(
        self,
        resampler,
        candidate_records: list[SearchRecord],
        visited_states: list[tuple[str, float, object]],
        terminal_indices: list[int],
        step: int,
        width: int,
    ):
        remaining_steps = self.num_steps - (step + 1)
        visited_states = [
            (identifier, value * self.backtrack, state)
            for identifier, value, state in visited_states
            if remaining_steps >= self.min_steps - len(state.steps)
        ]

        for i, record in enumerate(candidate_records):
            if i not in terminal_indices:
                visited_states.append((f"{i}.{step}", record.value, record.state))

        if not visited_states:
            return candidate_records, visited_states

        selected_indices = self._tournament_state_indices(visited_states, width)
        resampled_states = self._clone_selected_states(resampler, visited_states, selected_indices)
        records = [
            SearchRecord(
                state=resampled_state,
                value=float(visited_states[source_index][1]),
                depth=min(step + 1, self.num_steps - 1),
                uncertainty=0.0,
            )
            for resampled_state, source_index in zip(resampled_states, selected_indices)
        ]
        return records, visited_states
