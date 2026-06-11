from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from ..logging_utils import log_event
from .reagents import DifficultyAgentSpec, MethodReagents, StepAgentSpec


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

    def _base_probs(self, depth: int) -> np.ndarray:
        bounded_depth = max(0, min(depth, self.num_steps - 1))
        priors = np.nan_to_num(self.priors[bounded_depth], nan=0.0)
        total = priors.sum()
        if total <= 0:
            return np.ones(len(self.step_agents), dtype=float) / len(self.step_agents)
        probs = priors / total
        probs = np.maximum(probs, 1e-9)
        return probs / probs.sum()

    def _ranked_groups(self, probs: np.ndarray) -> tuple[list[int], list[int], dict[int, int]]:
        ascending_order = np.argsort(probs, kind="mergesort").tolist()
        ranks = {agent_index: rank + 1 for rank, agent_index in enumerate(ascending_order)}
        grouping = self.comparison_grouping.lower()

        if grouping == "even_odd":
            group_a = [agent_index for agent_index in ascending_order if ranks[agent_index] % 2 == 1]
            group_b = [agent_index for agent_index in ascending_order if ranks[agent_index] % 2 == 0]
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

    def _comparison_probs(self, depth: int) -> tuple[np.ndarray, list[int], list[int], list[float], list[int]]:
        probs = self._base_probs(depth)
        num_agents = len(self.step_agents)
        if num_agents == 1:
            return np.ones(1, dtype=float), [0], [], [1.0], [1]

        group_a, group_b, ranks = self._ranked_groups(probs)
        theta = max(0.0, min(1.0, self.comparison_theta))
        raw_probs = np.zeros(num_agents, dtype=float)

        for agent_index in range(num_agents):
            rank_i = ranks[agent_index]
            other_group = group_b if agent_index in group_a else group_a
            cross_group_sum = sum(num_agents - ranks[other_agent] for other_agent in other_group)

            rank_term = (2.0 * (rank_i - 1.0) / (num_agents * (num_agents - 1.0))) * theta
            comparison_term = (
                4.0
                * (1.0 - theta)
                * cross_group_sum
                / ((num_agents ** 2) * (num_agents - 1.0))
            )
            raw_probs[agent_index] = rank_term + comparison_term

        raw_total = raw_probs.sum()
        if raw_total <= 0:
            probs = np.ones(num_agents, dtype=float) / num_agents
        else:
            probs = raw_probs / raw_total
        epsilon = max(0.0, min(1.0, self.comparison_epsilon))
        probs = (1.0 - epsilon) * probs + epsilon / num_agents
        rank_vector = [ranks[agent_index] for agent_index in range(num_agents)]
        return probs / probs.sum(), group_a, group_b, raw_probs.tolist(), rank_vector

    def _sample_agent_index(self, depth: int) -> int:
        probs, group_a, group_b, raw_probs, ranks = self._comparison_probs(depth)
        selected = int(np.random.choice(len(self.step_agents), p=probs))
        log_event("REAGENTS_V2COMP_SELECTION", {
            "depth": depth,
            "selected_agent_index": selected,
            "group_a": group_a,
            "group_b": group_b,
            "ranks": ranks,
            "rrts_raw_probs": raw_probs,
            "comparison_probs": probs.tolist(),
        })
        return selected
