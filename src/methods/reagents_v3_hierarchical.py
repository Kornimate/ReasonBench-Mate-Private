import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..logging_utils import action_summary, log_event, log_section, log_section_end, terminal_summary
from ..typedefs import Agent, DecodingParameters, Environment, MAX_SEED, Model, State
from .reagents import DifficultyAgentSpec, MethodReagents, SearchRecord, StepAgentSpec
from .utils.reagents_common import ThompsonBandit, UCBBandit


@AgentDictFactory.register
class AgentDictReagents_v3_hierarchical(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@dataclass
class HierarchicalTrajectoryArm:
    record: SearchRecord
    active: bool = True
    history: list[dict] = field(default_factory=list)


@MethodFactory.register
class MethodReagents_v3_hierarchical(MethodReagents):
    """Hierarchical bandits over trajectories and step-agent operators."""

    def __init__(
        self,
        agents: AgentDictReagents_v3_hierarchical,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.trajectory_policy = str(getattr(config, "trajectory_policy", "thompson")).lower()
        self.operator_policy = str(getattr(config, "operator_policy", "ucb")).lower()
        self.bandit_num_arms = int(getattr(config, "bandit_num_arms", self.width))
        self.bandit_budget = int(getattr(config, "bandit_budget", self.width * self.num_steps))
        self.trajectory_reward_mode = str(getattr(config, "trajectory_reward_mode", "score")).lower()
        self.operator_reward_mode = str(getattr(config, "operator_reward_mode", "delta")).lower()
        self.reward_min = float(getattr(config, "bandit_reward_min", 0.0))
        self.reward_max = float(getattr(config, "bandit_reward_max", self.max_value))
        self.trajectory_discount = float(getattr(config, "trajectory_discount", 1.0))
        self.operator_discount = float(getattr(config, "operator_discount", 0.95))
        self.trajectory_ucb_c = float(getattr(config, "trajectory_ucb_c", 2.0 ** 0.5))
        self.operator_ucb_c = float(getattr(config, "operator_ucb_c", 2.0 ** 0.5))
        self.operator_min_count_per_type = int(getattr(config, "operator_min_count_per_type", 1))
        self.stop_on_solved = bool(getattr(config, "bandit_stop_on_solved", True))
        self.max_arm_depth = int(getattr(config, "bandit_max_arm_depth", 0))

        if self.bandit_budget < 1:
            raise ValueError("bandit_budget must be >= 1")
        if self.bandit_num_arms < 1:
            raise ValueError("bandit_num_arms must be >= 1")
        if self.reward_max <= self.reward_min:
            raise ValueError("bandit_reward_max must be greater than bandit_reward_min")
        if self.trajectory_policy not in {"thompson", "ucb"}:
            raise ValueError("trajectory_policy must be 'thompson' or 'ucb'")
        if self.operator_policy not in {"thompson", "ucb"}:
            raise ValueError("operator_policy must be 'thompson' or 'ucb'")
        if self.operator_min_count_per_type < 0:
            raise ValueError("operator_min_count_per_type must be >= 0")

        self.features["runtime_width_adaptation"] = False
        self.features["updating_priors"] = False

    def _make_bandit(self, policy: str, n_arms: int, discount: float, c: float):
        if policy == "ucb":
            return UCBBandit(n_arms=n_arms, c=c)
        return ThompsonBandit(n_arms=n_arms, discount=discount)

    def _bounded_reward(self, raw: float) -> float:
        scaled = (raw - self.reward_min) / (self.reward_max - self.reward_min)
        return float(max(0.0, min(1.0, scaled)))

    def _reward_from_values(self, old_value: float, new_value: float, mode: str) -> float:
        if mode == "score":
            return self._bounded_reward(float(new_value))
        if mode == "delta":
            return self._bounded_reward(max(0.0, float(new_value) - float(old_value)))
        raise ValueError("reward mode must be 'score' or 'delta'")

    def _active_indices(self, arms: list[HierarchicalTrajectoryArm]) -> list[int]:
        indices = []
        for i, arm in enumerate(arms):
            if not arm.active:
                continue
            if self.max_arm_depth > 0 and arm.record.depth >= self.max_arm_depth:
                continue
            indices.append(i)
        return indices

    def _select_operator(self, operator_bandit, pull: int) -> int:
        n_types = len(self.step_agents)
        forced_trials = n_types * self.operator_min_count_per_type
        if self.operator_min_count_per_type > 0 and pull < forced_trials:
            return pull % n_types
        return operator_bandit.select()

    async def _mutate_one_with_agent(
        self,
        record: SearchRecord,
        agent_index: int,
        namespace: str,
        idx: int,
        pull: int,
        arm_index: int,
    ):
        spec = self.step_agents[agent_index]
        action_batches = await asyncio.gather(
            spec["agent"].act(
                model=self.model,
                state=record.state,
                n=1,
                namespace=namespace,
                request_id=(
                    f"idx{idx}-pull{pull}-arm{arm_index}"
                    f"-{hash(record.state)}-agent{agent_index}"
                ),
                params=spec["params"],
            )
        )

        actions = action_batches[0]
        if not actions:
            new_state = record.state
        else:
            try:
                new_state = self.env.step(record.state, actions[0])
            except Exception:
                new_state = record.state

        return [
            SearchRecord(
                state=new_state,
                value=record.value,
                depth=record.depth + 1,
            )
        ], action_batches

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        random.seed(idx)
        np.random.seed(idx)

        width = await self._get_width(state, idx)
        num_arms = int(getattr(self.config, "bandit_num_arms", width))
        budget = max(num_arms, self.bandit_budget)
        trajectory_bandit = self._make_bandit(
            self.trajectory_policy,
            num_arms,
            self.trajectory_discount,
            self.trajectory_ucb_c,
        )
        operator_bandit = self._make_bandit(
            self.operator_policy,
            len(self.step_agents),
            self.operator_discount,
            self.operator_ucb_c,
        )

        initial_value = self.origin * self.backtrack * self.backtrack
        arms = [
            HierarchicalTrajectoryArm(
                record=SearchRecord(
                    state=state.clone(randomness=random.randint(0, MAX_SEED)),
                    value=initial_value,
                    depth=0,
                )
            )
            for _ in range(num_arms)
        ]

        log_section("ReAgents v3 Hierarchical / Trajectory + Agent-Type Allocation:")
        log_event(
            "REAGENTS_V3_HIERARCHICAL_START",
            {
                "idx": idx,
                "trajectory_policy": self.trajectory_policy,
                "operator_policy": self.operator_policy,
                "num_arms": num_arms,
                "budget": budget,
                "trajectory_reward_mode": self.trajectory_reward_mode,
                "operator_reward_mode": self.operator_reward_mode,
                "reward_min": self.reward_min,
                "reward_max": self.reward_max,
                "operator_min_count_per_type": self.operator_min_count_per_type,
            },
        )

        solved_records: list[SearchRecord] = []
        for pull in range(budget):
            active = self._active_indices(arms)
            selected_arm = trajectory_bandit.select(active)
            if selected_arm < 0:
                break

            selected_operator = self._select_operator(operator_bandit, pull)
            old_record = arms[selected_arm].record
            new_records, action_batches = await self._mutate_one_with_agent(
                record=old_record,
                agent_index=selected_operator,
                namespace=namespace,
                idx=idx,
                pull=pull,
                arm_index=selected_arm,
            )
            new_records, terminal_indices, solved_indices = await self._evaluate_states(
                new_records, value_cache, namespace, idx, pull
            )
            new_record = new_records[0]

            trajectory_reward = self._reward_from_values(
                old_record.value, new_record.value, self.trajectory_reward_mode
            )
            operator_reward = self._reward_from_values(
                old_record.value, new_record.value, self.operator_reward_mode
            )
            trajectory_bandit.update(selected_arm, trajectory_reward)
            operator_bandit.update(selected_operator, operator_reward)
            arms[selected_arm].record = new_record

            try:
                is_final, final_score = self.env.evaluate(new_record.state)
            except Exception:
                is_final, final_score = False, 0.0
            solved = bool(solved_indices) or self._normalize_score(final_score) == 1.0
            if is_final or solved:
                arms[selected_arm].active = False

            agent_type = self.step_agents[selected_operator].get(
                "agent_type", f"agent_{selected_operator}"
            )
            arms[selected_arm].history.append(
                {
                    "pull": pull,
                    "selected_arm": selected_arm,
                    "selected_operator": selected_operator,
                    "agent_type": agent_type,
                    "old_value": old_record.value,
                    "new_value": new_record.value,
                    "trajectory_reward": trajectory_reward,
                    "operator_reward": operator_reward,
                }
            )

            log_event(
                "REAGENTS_V3_HIERARCHICAL_PULL",
                {
                    "idx": idx,
                    "pull": pull,
                    "selected_arm": selected_arm,
                    "selected_operator": selected_operator,
                    "agent_type": agent_type,
                    "old_value": old_record.value,
                    "new_value": new_record.value,
                    "trajectory_reward": trajectory_reward,
                    "operator_reward": operator_reward,
                    "new_depth": new_record.depth,
                    "actions": action_summary(action_batches),
                    "terminal_indices": terminal_indices,
                    "solved_indices": solved_indices,
                    "trajectory_bandit": trajectory_bandit.snapshot(),
                    "operator_bandit": operator_bandit.snapshot(),
                    "terminal": terminal_summary(self.env, [new_record.state]),
                },
            )

            if solved:
                solved_records.append(new_record)
                if self.stop_on_solved:
                    self._log_final_records(
                        idx,
                        solved_records,
                        "REAGENTS_V3_HIERARCHICAL_FINAL_SOLVED",
                    )
                    log_section_end()
                    return [record.state for record in solved_records]

        final_records = sorted([arm.record for arm in arms], key=lambda r: r.value, reverse=True)
        if solved_records:
            self._log_final_records(
                idx,
                solved_records,
                "REAGENTS_V3_HIERARCHICAL_FINAL_SOLVED",
            )
            log_section_end()
            return [record.state for record in solved_records]

        self._log_final_records(idx, final_records, "REAGENTS_V3_HIERARCHICAL_FINAL_RECORDS")
        log_section_end()
        return [record.state for record in final_records] if final_records else [state]
