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
class AgentDictReagents_v3_base(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@dataclass
class TrajectoryArm:
    record: SearchRecord
    active: bool = True
    history: list[dict] = field(default_factory=list)


@MethodFactory.register
class MethodReagents_v3_base(MethodReagents):
    """BaSE-style allocation over parallel ReAgents trajectories."""

    def __init__(
        self,
        agents: AgentDictReagents_v3_base,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.trajectory_policy = str(getattr(config, "trajectory_policy", "thompson")).lower()
        self.bandit_num_arms = int(getattr(config, "bandit_num_arms", self.width))
        self.bandit_budget = int(getattr(config, "bandit_budget", self.width * self.num_steps))
        self.bandit_reward_mode = str(getattr(config, "bandit_reward_mode", "score")).lower()
        self.bandit_reward_min = float(getattr(config, "bandit_reward_min", 0.0))
        self.bandit_reward_max = float(getattr(config, "bandit_reward_max", self.max_value))
        self.trajectory_discount = float(getattr(config, "trajectory_discount", 1.0))
        self.trajectory_ucb_c = float(getattr(config, "trajectory_ucb_c", 2.0 ** 0.5))
        self.stop_on_solved = bool(getattr(config, "bandit_stop_on_solved", True))
        self.max_arm_depth = int(getattr(config, "bandit_max_arm_depth", 0))

        if self.bandit_budget < 1:
            raise ValueError("bandit_budget must be >= 1")
        if self.bandit_num_arms < 1:
            raise ValueError("bandit_num_arms must be >= 1")
        if self.bandit_reward_max <= self.bandit_reward_min:
            raise ValueError("bandit_reward_max must be greater than bandit_reward_min")
        if self.trajectory_policy not in {"thompson", "ucb"}:
            raise ValueError("trajectory_policy must be 'thompson' or 'ucb'")

        self.features["runtime_width_adaptation"] = False

    def _make_trajectory_bandit(self, n_arms: int):
        if self.trajectory_policy == "ucb":
            return UCBBandit(n_arms=n_arms, c=self.trajectory_ucb_c)
        return ThompsonBandit(n_arms=n_arms, discount=self.trajectory_discount)

    def _reward_from_values(self, old_value: float, new_value: float) -> float:
        if self.bandit_reward_mode == "delta":
            raw = max(0.0, float(new_value) - float(old_value))
        elif self.bandit_reward_mode == "score":
            raw = float(new_value)
        else:
            raise ValueError("bandit_reward_mode must be 'score' or 'delta'")

        scaled = (raw - self.bandit_reward_min) / (self.bandit_reward_max - self.bandit_reward_min)
        return float(max(0.0, min(1.0, scaled)))

    def _active_indices(self, arms: list[TrajectoryArm]) -> list[int]:
        active = []
        for i, arm in enumerate(arms):
            if not arm.active:
                continue
            if self.max_arm_depth > 0 and arm.record.depth >= self.max_arm_depth:
                continue
            active.append(i)
        return active

    async def _pull_trajectory(
        self,
        arm: TrajectoryArm,
        arm_index: int,
        namespace: str,
        idx: int,
        pull: int,
        value_cache: dict | None,
    ):
        old_record = arm.record
        new_records, agent_indices, action_batches = await self._mutate_states(
            [old_record], namespace, idx, pull
        )
        new_records, terminal_indices, solved_indices = await self._evaluate_states(
            new_records, value_cache, namespace, idx, pull
        )

        new_record = new_records[0]
        reward = self._reward_from_values(old_record.value, new_record.value)
        self._update_priors(agent_indices, [old_record], [new_record])
        arm.record = new_record

        try:
            is_final, final_score = self.env.evaluate(new_record.state)
        except Exception:
            is_final, final_score = False, 0.0
        solved = bool(solved_indices) or self._normalize_score(final_score) == 1.0
        if is_final or solved:
            arm.active = False

        agent_types = [self.step_agents[i].get("agent_type", f"agent_{i}") for i in agent_indices]
        arm.history.append(
            {
                "pull": pull,
                "arm": arm_index,
                "old_value": old_record.value,
                "new_value": new_record.value,
                "reward": reward,
                "agent_indices": agent_indices,
                "agent_types": agent_types,
                "actions": action_summary(action_batches),
                "terminal_indices": terminal_indices,
                "solved_indices": solved_indices,
            }
        )

        return (
            new_record,
            reward,
            agent_indices,
            agent_types,
            action_batches,
            terminal_indices,
            solved_indices,
            solved,
        )

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        random.seed(idx)
        np.random.seed(idx)

        width = await self._get_width(state, idx)
        num_arms = int(getattr(self.config, "bandit_num_arms", width))
        budget = max(num_arms, self.bandit_budget)
        bandit = self._make_trajectory_bandit(num_arms)
        initial_value = self.origin * self.backtrack * self.backtrack

        arms = [
            TrajectoryArm(
                record=SearchRecord(
                    state=state.clone(randomness=random.randint(0, MAX_SEED)),
                    value=initial_value,
                    depth=0,
                )
            )
            for _ in range(num_arms)
        ]

        log_section("ReAgents v3 Base / BaSE Trajectory Allocation:")
        log_event(
            "REAGENTS_V3_BASE_START",
            {
                "idx": idx,
                "trajectory_policy": self.trajectory_policy,
                "num_arms": num_arms,
                "budget": budget,
                "reward_mode": self.bandit_reward_mode,
                "reward_min": self.bandit_reward_min,
                "reward_max": self.bandit_reward_max,
                "discount": self.trajectory_discount,
            },
        )

        solved_records: list[SearchRecord] = []
        for pull in range(budget):
            active = self._active_indices(arms)
            selected = bandit.select(active)
            if selected < 0:
                break

            (
                new_record,
                reward,
                agent_indices,
                agent_types,
                action_batches,
                terminal_indices,
                solved_indices,
                solved,
            ) = await self._pull_trajectory(
                arm=arms[selected],
                arm_index=selected,
                namespace=namespace,
                idx=idx,
                pull=pull,
                value_cache=value_cache,
            )
            bandit.update(selected, reward)

            log_event(
                "REAGENTS_V3_BASE_PULL",
                {
                    "idx": idx,
                    "pull": pull,
                    "selected_arm": selected,
                    "reward": reward,
                    "new_value": new_record.value,
                    "new_depth": new_record.depth,
                    "agent_indices": agent_indices,
                    "agent_types": agent_types,
                    "actions": action_summary(action_batches),
                    "terminal_indices": terminal_indices,
                    "solved_indices": solved_indices,
                    "bandit": bandit.snapshot(),
                    "terminal": terminal_summary(self.env, [new_record.state]),
                },
            )

            if solved:
                solved_records.append(new_record)
                if self.stop_on_solved:
                    self._log_final_records(idx, solved_records, "REAGENTS_V3_BASE_FINAL_SOLVED")
                    log_section_end()
                    return [record.state for record in solved_records]

        final_records = sorted([arm.record for arm in arms], key=lambda r: r.value, reverse=True)
        if solved_records:
            self._log_final_records(idx, solved_records, "REAGENTS_V3_BASE_FINAL_SOLVED")
            log_section_end()
            return [record.state for record in solved_records]

        self._log_final_records(idx, final_records, "REAGENTS_V3_BASE_FINAL_RECORDS")
        log_section_end()
        return [record.state for record in final_records] if final_records else [state]
