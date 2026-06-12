import random
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from ..logging_utils import log_event
from .reagents import DifficultyAgentSpec, MethodReagents, SearchRecord, StepAgentSpec


@AgentDictFactory.register
class AgentDictReagents_v2comp(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v2comp(MethodReagents):
    def __init__(
        self,
        agents: AgentDictReagents_v2comp,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.comparison_theta = float(getattr(config, "comparison_theta", 0.5))
        self.comparison_epsilon = float(getattr(config, "comparison_epsilon", 0.0))
        self.comparison_grouping = str(getattr(config, "comparison_grouping", "random"))

    def _ranked_groups(self, values: list[float]) -> tuple[list[int], list[int], dict[int, int]]:
        ascending_order = np.argsort(values, kind="mergesort").tolist()
        ranks = {record_index: rank + 1 for rank, record_index in enumerate(ascending_order)}
        grouping = self.comparison_grouping.lower()

        if grouping == "even_odd":
            group_a = [record_index for record_index in ascending_order if ranks[record_index] % 2 == 1]
            group_b = [record_index for record_index in ascending_order if ranks[record_index] % 2 == 0]
        elif grouping == "best_worst":
            midpoint = max(1, len(ascending_order) // 2)
            group_a = ascending_order[:midpoint]
            group_b = ascending_order[midpoint:]
        else:
            shuffled = np.random.permutation(len(self.step_agents)).tolist()
            midpoint = max(1, len(shuffled) // 2)
            group_a = shuffled[:midpoint]
            group_b = shuffled[midpoint:]

        if not group_b:
            group_b = [group_a.pop()]
        return group_a, group_b, ranks

    def _comparison_probs(self, visited_states) -> tuple[np.ndarray, list[int], list[int], list[float], list[int]]:
        num_records = len(visited_states)
        if num_records == 1:
            return np.ones(1, dtype=float), [0], [], [1.0], [1]

        values = [float(value) for _, value, _ in visited_states]
        group_a, group_b, ranks = self._ranked_groups(values)
        theta = max(0.0, min(1.0, self.comparison_theta))
        raw_probs = np.zeros(num_records, dtype=float)

        for record_index in range(num_records):
            rank_i = ranks[record_index]
            other_group = group_b if record_index in group_a else group_a
            cross_group_sum = sum(num_records - ranks[other_record] for other_record in other_group)

            rank_term = (2.0 * (rank_i - 1.0) / (num_records * (num_records - 1.0))) * theta
            comparison_term = (
                4.0
                * (1.0 - theta)
                * cross_group_sum
                / ((num_records ** 2) * (num_records - 1.0))
            )
            raw_probs[record_index] = rank_term + comparison_term

        raw_total = raw_probs.sum()
        if raw_total <= 0:
            probs = np.ones(num_records, dtype=float) / num_records
        else:
            probs = raw_probs / raw_total
        epsilon = max(0.0, min(1.0, self.comparison_epsilon))
        probs = (1.0 - epsilon) * probs + epsilon / num_records
        rank_vector = [ranks[record_index] for record_index in range(num_records)]
        return probs / probs.sum(), group_a, group_b, raw_probs.tolist(), rank_vector

    def _clone_selected_states(self, resampler, visited_states, selected_indices):
        random.seed(resampler.randomness)
        new_randomness = [random.randint(1, 1000) for _ in selected_indices]
        if new_randomness:
            resampler.randomness = new_randomness[-1]
        return [
            visited_states[source_index][2].clone(randomness)
            for source_index, randomness in zip(selected_indices, new_randomness)
        ]

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

        probs, group_a, group_b, raw_probs, ranks = self._comparison_probs(visited_states)
        selected_indices = np.random.choice(
            len(visited_states),
            size=width,
            replace=True,
            p=probs,
        ).tolist()

        log_event("REAGENTS_V2COMP_STATE_SELECTION", {
            "width": width,
            "selected_indices": selected_indices,
            "group_a": group_a,
            "group_b": group_b,
            "ranks": ranks,
            "rrts_raw_probs": raw_probs,
            "comparison_probs": probs.tolist(),
        })

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
