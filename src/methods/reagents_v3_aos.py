import asyncio
import random
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..logging_utils import (
    action_summary,
    log_event,
    log_section,
    log_section_end,
    score_summary,
    terminal_summary,
)
from ..typedefs import Agent, DecodingParameters, Environment, MAX_SEED, Model, State
from ..utils import Resampler
from .reagents import DifficultyAgentSpec, MethodReagents, SearchRecord, StepAgentSpec
from .utils.reagents_common import ThompsonBandit, UCBBandit


@AgentDictFactory.register
class AgentDictReagents_v3_aos(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v3_aos(MethodReagents):
    """Adaptive Operator Selection over ReAgents step-agent types."""

    def __init__(
        self,
        agents: AgentDictReagents_v3_aos,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.operator_policy = str(getattr(config, "operator_policy", "ucb")).lower()
        self.operator_reward_mode = str(getattr(config, "operator_reward_mode", "delta")).lower()
        self.operator_reward_min = float(getattr(config, "operator_reward_min", 0.0))
        self.operator_reward_max = float(getattr(config, "operator_reward_max", self.max_value))
        self.operator_discount = float(getattr(config, "operator_discount", 0.95))
        self.operator_ucb_c = float(getattr(config, "operator_ucb_c", 2.0 ** 0.5))
        self.operator_min_count_per_type = int(getattr(config, "operator_min_count_per_type", 1))

        if self.operator_policy not in {"thompson", "ucb"}:
            raise ValueError("operator_policy must be 'thompson' or 'ucb'")
        if self.operator_reward_max <= self.operator_reward_min:
            raise ValueError("operator_reward_max must be greater than operator_reward_min")
        if self.operator_min_count_per_type < 0:
            raise ValueError("operator_min_count_per_type must be >= 0")

        self.features["updating_priors"] = False

    def _make_operator_bandit(self):
        if self.operator_policy == "ucb":
            return UCBBandit(n_arms=len(self.step_agents), c=self.operator_ucb_c)
        return ThompsonBandit(n_arms=len(self.step_agents), discount=self.operator_discount)

    def _operator_reward(self, old_value: float, new_value: float) -> float:
        if self.operator_reward_mode == "score":
            raw = float(new_value)
        elif self.operator_reward_mode == "delta":
            raw = max(0.0, float(new_value) - float(old_value))
        else:
            raise ValueError("operator_reward_mode must be 'score' or 'delta'")

        scaled = (raw - self.operator_reward_min) / (
            self.operator_reward_max - self.operator_reward_min
        )
        return float(max(0.0, min(1.0, scaled)))

    def _allocate_agent_indices(self, width: int, operator_bandit) -> list[int]:
        if width <= 0:
            return []

        indices: list[int] = []
        n_types = len(self.step_agents)
        required = n_types * self.operator_min_count_per_type

        if self.operator_min_count_per_type > 0 and width >= required:
            for agent_index in range(n_types):
                indices.extend([agent_index] * self.operator_min_count_per_type)

        while len(indices) < width:
            indices.append(operator_bandit.select())

        indices = indices[:width]
        random.shuffle(indices)
        return indices

    async def _mutate_states(
        self,
        records: list[SearchRecord],
        namespace: str,
        idx: int,
        step: int,
        operator_bandit,
    ):
        agent_indices = self._allocate_agent_indices(len(records), operator_bandit)
        coroutines = []

        for i, (record, agent_index) in enumerate(zip(records, agent_indices)):
            spec = self.step_agents[agent_index]
            coroutines.append(
                spec["agent"].act(
                    model=self.model,
                    state=record.state,
                    n=1,
                    namespace=namespace,
                    request_id=(
                        f"idx{idx}-step{step}-{hash(record.state)}"
                        f"-aos-agent{agent_index + 100 * i}"
                    ),
                    params=spec["params"],
                )
            )

        action_batches = await asyncio.gather(*coroutines) if coroutines else []
        new_records = []
        for record, actions in zip(records, action_batches):
            if not actions:
                new_records.append(
                    SearchRecord(state=record.state, value=record.value, depth=record.depth + 1)
                )
                continue
            try:
                new_state = self.env.step(record.state, actions[0])
            except Exception:
                new_state = record.state
            new_records.append(SearchRecord(state=new_state, value=record.value, depth=record.depth + 1))

        return new_records, agent_indices, action_batches

    def _update_operator_bandit(
        self,
        old_records: list[SearchRecord],
        new_records: list[SearchRecord],
        agent_indices: list[int],
        operator_bandit,
    ) -> list[float]:
        rewards = []
        for old, new, agent_index in zip(old_records, new_records, agent_indices):
            reward = self._operator_reward(old.value, new.value)
            operator_bandit.update(agent_index, reward)
            rewards.append(reward)
        return rewards

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        random.seed(idx)
        np.random.seed(idx)
        resampler = Resampler(idx)
        operator_bandit = self._make_operator_bandit()

        width = await self._get_width(state, idx)
        initial_value = self.origin * self.backtrack * self.backtrack
        records = [
            SearchRecord(
                state=state.clone(randomness=random.randint(0, MAX_SEED)),
                value=initial_value,
                depth=0,
            )
            for _ in range(width)
        ]
        visited_states = [("INIT", self.origin, state)]

        log_section("ReAgents v3 AOS / Agent-Type Allocation:")
        log_event(
            "REAGENTS_V3_AOS_START",
            {
                "idx": idx,
                "operator_policy": self.operator_policy,
                "reward_mode": self.operator_reward_mode,
                "reward_min": self.operator_reward_min,
                "reward_max": self.operator_reward_max,
                "discount": self.operator_discount,
                "operator_min_count_per_type": self.operator_min_count_per_type,
            },
        )

        for step in range(self.num_steps):
            old_width = width
            old_records = records
            new_records, agent_indices, action_batches = await self._mutate_states(
                records, namespace, idx, step, operator_bandit
            )
            new_records, terminal_indices, solved_indices = await self._evaluate_states(
                new_records, value_cache, namespace, idx, step
            )

            rewards = self._update_operator_bandit(
                old_records, new_records, agent_indices, operator_bandit
            )
            width = self._update_width(old_records, new_records, width)
            agent_types = [self.step_agents[i].get("agent_type", f"agent_{i}") for i in agent_indices]

            log_event(
                "REAGENTS_V3_AOS_STEP",
                {
                    "idx": idx,
                    "step": step,
                    "width": old_width,
                    "next_width": width,
                    "agent_indices": agent_indices,
                    "agent_types": agent_types,
                    "agent_counts": {
                        agent_type: agent_types.count(agent_type)
                        for agent_type in sorted(set(agent_types))
                    },
                    "rewards": score_summary(rewards),
                    "old_values": score_summary(record.value for record in old_records),
                    "new_values": score_summary(record.value for record in new_records),
                    "operator_bandit": operator_bandit.snapshot(),
                    "actions": action_summary(action_batches),
                    "terminal_indices": terminal_indices,
                    "solved_indices": solved_indices,
                    "terminal": terminal_summary(self.env, [record.state for record in new_records]),
                },
            )

            if solved_indices:
                solved_records = [new_records[i] for i in solved_indices]
                self._log_final_records(idx, solved_records, "REAGENTS_V3_AOS_FINAL_SOLVED")
                log_section_end()
                return [record.state for record in solved_records]

            new_records, visited_states = self._filter_states(
                old_records, new_records, visited_states
            )
            records, visited_states = self._resample_records(
                resampler,
                new_records,
                visited_states,
                terminal_indices,
                step,
                width,
            )
            if not records:
                break

        self._log_final_records(idx, records, "REAGENTS_V3_AOS_FINAL_RECORDS")
        log_section_end()
        return [record.state for record in records] if records else [state]
