import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, MAX_SEED, Method, Model, State
from ..utils import Resampler
from .new_algo import DifficultyAgentSpec, SearchRecord

logger = logging.getLogger("__main__")


class StepAgentInfo(TypedDict):
    agent_type: str
    agent: Agent
    params: DecodingParameters
    num_agents: int

@dataclass
class AOSAllocator:
    num_agents_types: int
    num_steps: int
    alpha: float = 0.2
    temperature: float = 0.25
    epsilon: float = 0.10
    min_count_per_type: int = 0

    q_values: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.q_values = np.zeros((self.num_steps, self.num_agents_types), dtype=float)
        self.counts = np.zeros((self.num_steps, self.num_agents_types), dtype=int)

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        temperature = max(self.temperature, 1e-6)
        centered = scores - np.max(scores)
        exp_scores = np.exp(centered / temperature)
        probs = exp_scores / np.sum(exp_scores)

        # exploration floor
        k = len(probs)
        probs = (1.0 - self.epsilon) * probs + self.epsilon / k
        probs = probs / probs.sum()

        return probs

    def get_probs(self, depth: int) -> np.ndarray:
        depth = max(0, min(depth, self.num_steps - 1))
        return self._softmax(self.q_values[depth])

        
    # return exact integer counts for each operator.
    def allocate(self, width: int, depth: int) -> list[int]:
        probs = self.get_probs(depth)
        raw = width * probs

        counts = np.floor(raw).astype(int)
        remainder = width - int(counts.sum())

        fractional = raw - counts
        order = np.argsort(-fractional)

        for idx in order[:remainder]:
            counts[idx] += 1

        # optional minimum count rule, want both act/react present
        if self.min_count_per_type > 0 and width >= self.num_agents_types * self.min_count_per_type:
            for i in range(self.num_agents_types):
                if counts[i] < self.min_count_per_type:
                    donor = int(np.argmax(counts))
                    if counts[donor] > self.min_count_per_type:
                        counts[donor] -= 1
                        counts[i] += 1

        return counts.tolist()

    # update Q-values from observed rewards.
    def update(self, depth: int, agent_indices: list[int], rewards: list[float]) -> None:
        if not agent_indices:
            return

        depth = max(0, min(depth, self.num_steps - 1))

        by_agent: dict[int, list[float]] = {}
        for agent_idx, reward in zip(agent_indices, rewards):
            by_agent.setdefault(agent_idx, []).append(float(reward))

        for agent_idx, rs in by_agent.items():
            avg_reward = float(np.mean(rs))
            old_q = self.q_values[depth, agent_idx]
            self.q_values[depth, agent_idx] = (
                (1.0 - self.alpha) * old_q + self.alpha * avg_reward
            )
            self.counts[depth, agent_idx] += len(rs)


@AgentDictFactory.register
class AgentDictReagentsAOS(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentInfo]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagentsAOS(Method):
    def __init__(
        self,
        agents: AgentDictReagentsAOS,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(model, agents, env, config)

        self.eval_agent = agents["evaluate"]
        self.evaluate_params = agents["evaluate_params"]
        self.step_agents = agents["step_agents"]
        self.difficulty_agent = agents.get("difficulty_agent")

        self.width = int(config.width)
        self.num_steps = int(config.num_steps)
        self.max_value = float(config.max_value)
        self.k = int(config.k)
        self.alpha = float(config.alpha)
        self.backtrack = float(config.backtrack)
        self.resampling = str(config.resampling)
        self.origin = float(config.origin)
        self.min_steps = int(config.min_steps)
        self.num_evaluations = int(config.num_evaluations)

        self.features = {
            "updating_priors": bool(getattr(config, "updating_priors", True)),
            "difficulty_based_width_init": bool(getattr(config, "difficulty_based_width_init", True)),
            "runtime_width_adaptation": bool(getattr(config, "runtime_width_adaptation", True)),
            "skewed_state_detection": bool(getattr(config, "skewed_state_detection", False)),
        }

        self.priors = np.ones((self.num_steps, len(self.step_agents))) / max(len(self.step_agents), 1)

        # step_agents[0] = act
        # step_agents[1] = react
        self.allocator = AOSAllocator(
            num_agents_types=len(self.step_agents),
            num_steps=self.num_steps,
            alpha=float(getattr(config, "aos_alpha", 0.2)),
            temperature=float(getattr(config, "aos_temperature", 0.25)),
            epsilon=float(getattr(config, "aos_epsilon", 0.10)),
            min_count_per_type=int(getattr(config, "aos_min_count_per_type", 0)),
        )

    def _normalize_score(self, score) -> float:
        if isinstance(score, (int, float)):
            return float(score)
        if isinstance(score, tuple) and score:
            return self._normalize_score(score[-1])
        if isinstance(score, list) and score:
            return self._normalize_score(score[-1])
        return 0.0

    def _sample_agent_index(self, depth: int) -> int:
        bounded_depth = max(0, min(depth, self.num_steps - 1))
        probs = np.nan_to_num(self.priors[bounded_depth], nan=0.0)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(self.step_agents)) / len(self.step_agents)
        else:
            probs = probs / total
        return int(np.random.choice(len(self.step_agents), p=probs))

    async def _get_width(self, state: State, idx: int) -> int:
        if not self.difficulty_agent or not self.features["difficulty_based_width_init"]:
            return self.width

        rating = await self.difficulty_agent["agent"].act(
            model=self.model,
            state=state,
            max_agents=self.width,
            namespace=f"diff-{idx}",
            request_id=f"idx{idx}-difficulty",
            params=self.difficulty_agent["params"],
        )
        try:
            rating_int = int(rating)
        except Exception:
            return self.width

        width_map = {
            1: max(1, self.width - self.width // 2),
            3: self.width,
            5: self.width + self.width // 2,
        }
        return width_map.get(rating_int, self.width)

    async def _mutate_states(
        self,
        records: list[SearchRecord],
        namespace: str,
        idx: int,
        step: int,
    ):
        width = len(records)

        # exact fleet allocation
        counts = self.allocator.allocate(width=width, depth=step)

        agent_indices = []
        for agent_index, count in enumerate(counts):
            agent_indices.extend([agent_index] * count)

        # Safety in case rounding/min-count logic somehow mismatches
        agent_indices = agent_indices[:width]
        while len(agent_indices) < width:
            agent_indices.append(int(np.argmax(self.allocator.get_probs(step))))

        random.shuffle(agent_indices)

        coroutines = []

        for i, record in enumerate(records):
            agent_index = agent_indices[i]
            spec = self.step_agents[agent_index]

            coroutines.append(
                spec["agent"].act(
                    model=self.model,
                    state=record.state,
                    n=1,
                    namespace=namespace,
                    request_id=(
                        f"idx{idx}-step{step}-"
                        f"{hash(record.state)}-agent{agent_index + 100 * i}"
                    ),
                    params=spec["params"],
                )
            )

        action_batches = await asyncio.gather(*coroutines)

        new_records = []
        for record, actions in zip(records, action_batches):
            if not actions:
                new_records.append(
                    SearchRecord(
                        state=record.state,
                        value=record.value,
                        depth=record.depth + 1,
                    )
                )
                continue

            try:
                new_state = self.env.step(record.state, actions[0])
            except Exception:
                new_state = record.state

            new_records.append(
                SearchRecord(
                    state=new_state,
                    value=record.value,
                    depth=record.depth + 1,
                )
            )

        return new_records, agent_indices, counts

    async def _evaluate_states(
        self,
        records: list[SearchRecord],
        value_cache: dict | None,
        namespace: str,
        idx: int,
        step: int,
    ):
        terminal_indices = []
        solved_indices = []
        coroutines = []
        pending_indices = []

        for i, record in enumerate(records):
            if self.env.is_final(record.state):
                terminal_indices.append(i)
            if self.env.evaluate(record.state)[1] == 1:
                solved_indices.append(i)
                terminal_indices.append(i)
                continue
            if i not in terminal_indices:
                pending_indices.append(i)
                coroutines.append(
                    self.eval_agent.act(
                        model=self.model,
                        state=record.state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{step}-{hash(record.state)}-agent{i}",
                        params=self.evaluate_params,
                        cache=value_cache,
                    )
                )

        evaluated = await asyncio.gather(*coroutines) if coroutines else []
        score_map = {record_index: self._normalize_score(score) for record_index, score in zip(pending_indices, evaluated)}

        updated_records = []
        for i, record in enumerate(records):
            if i in solved_indices:
                updated_records.append(SearchRecord(record.state, self.max_value, record.depth, 0.0))
            elif i in terminal_indices:
                updated_records.append(SearchRecord(record.state, 0.0, record.depth, 0.0))
            else:
                updated_records.append(SearchRecord(record.state, score_map.get(i, record.value), record.depth, 0.0))

        return updated_records, sorted(set(terminal_indices)), solved_indices

    def _update_width(self, old_records: list[SearchRecord], new_records: list[SearchRecord], width: int) -> int:
        if not self.features["runtime_width_adaptation"] or not old_records or not new_records:
            return width

        old_avg = sum(record.value for record in old_records) / len(old_records)
        new_avg = sum(record.value for record in new_records) / len(new_records)

        if old_avg < new_avg:
            return max(max(1, self.width // 2), width - 1)
        return min(self.width + self.width // 2, width + 1)

    def _average_reward_for_agent(
        self,
        agent_indices: list[int],
        rewards: list[float],
        agent_index: int,
    ) -> Optional[float]:
        agent_rewards = [
            float(reward)
            for assigned_agent_index, reward in zip(agent_indices, rewards)
            if assigned_agent_index == agent_index
        ]
        if not agent_rewards:
            return None
        return float(np.mean(agent_rewards))

    def _round_log_value(self, value):
        if isinstance(value, dict):
            return {
                key: self._round_log_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._round_log_value(item) for item in value]
        if isinstance(value, float):
            return round(value, 4)
        return value

    def _log_aos_step(
        self,
        step: int,
        width: int,
        fleet_counts: list[int],
        agent_indices: list[int],
        rewards: list[float],
        solved_indices: list[int],
    ) -> None:
        agent_types = [
            spec.get("agent_type", f"agent_{agent_index}")
            for agent_index, spec in enumerate(self.step_agents)
        ]
        agent_counts = {
            agent_type: int(fleet_counts[agent_index])
            for agent_index, agent_type in enumerate(agent_types)
        }
        agent_distribution = {
            agent_type: (float(count) / width if width else 0.0)
            for agent_type, count in agent_counts.items()
        }
        avg_reward_by_agent = {
            agent_type: self._average_reward_for_agent(agent_indices, rewards, agent_index)
            for agent_index, agent_type in enumerate(agent_types)
        }
        log_entry = {
            "step": step,
            "width": width,
            "agent_types": agent_types,
            "fleet_counts": fleet_counts,
            "agent_counts": agent_counts,
            "agent_distribution": agent_distribution,
            "q_values": self.allocator.q_values[step].tolist(),
            "probs": self.allocator.get_probs(step).tolist(),
            "avg_reward_by_agent": avg_reward_by_agent,
            "avg_reward_act": avg_reward_by_agent.get("act"),
            "avg_reward_react": avg_reward_by_agent.get("react"),
            "solved": bool(solved_indices),
        }
        logger.info("\tAOS_STEP %s", json.dumps(self._round_log_value(log_entry)))

    def _filter_states(
        self,
        old_records: list[SearchRecord],
        new_records: list[SearchRecord],
        visited: list[tuple[str, float, State]],
    ):
        if not self.features["skewed_state_detection"]:
            return new_records, visited

        grouped: dict[str, list[int]] = {}
        for i, record in enumerate(old_records):
            key = str(record.state.serialize().get("current_state", record.state.serialize()))
            grouped.setdefault(key, []).append(i)

        skewed_keys = set()
        for key, indices in grouped.items():
            if len(indices) <= 1:
                continue
            parent_avg = sum(old_records[i].value for i in indices) / len(indices)
            children = [new_records[i] for i in indices]
            if all(child.value <= parent_avg / 2 for child in children):
                skewed_keys.add(key)

        if not skewed_keys:
            return new_records, visited

        filtered_records = [
            record
            for record in new_records
            if str(record.state.serialize().get("current_state", record.state.serialize())) not in skewed_keys
        ]
        filtered_visited = [
            item
            for item in visited
            if str(item[2].serialize().get("current_state", item[2].serialize())) not in skewed_keys
        ]
        return filtered_records, filtered_visited

    def _resample_records(
        self,
        resampler: Resampler,
        candidate_records: list[SearchRecord],
        visited_states: list[tuple[str, float, State]],
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

        resampled_states, resampled_indices = resampler.resample(visited_states, width, self.resampling)
        records = [
            SearchRecord(
                state=resampled_state,
                value=float(visited_states[source_index][1]),
                depth=min(step + 1, self.num_steps - 1),
                uncertainty=0.0,
            )
            for resampled_state, source_index in zip(resampled_states, resampled_indices)
        ]
        return records, visited_states

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        random.seed(idx)
        np.random.seed(idx)
        resampler = Resampler(idx)

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
        visited_states: list[tuple[str, float, State]] = [("INIT", self.origin, state)]

        logger.info("Runtime Agent Distribution Information:")

        for step in range(self.num_steps):
            new_records, agent_indices, fleet_counts = await self._mutate_states(
                records, namespace, idx, step
            )

            new_records, terminal_indices, solved_indices = await self._evaluate_states(
                new_records, value_cache, namespace, idx, step
            )

            rewards = [
                max(-1.0, min(1.0, new.value - old.value))
                for old, new in zip(records, new_records)
            ]

            self.allocator.update(
                depth=step,
                agent_indices=agent_indices,
                rewards=rewards,
            )

            self._log_aos_step(
                step=step,
                width=width,
                fleet_counts=fleet_counts,
                agent_indices=agent_indices,
                rewards=rewards,
                solved_indices=solved_indices,
            )

            width = self._update_width(records, new_records, width)

            if solved_indices:
                return [new_records[i].state for i in solved_indices]

            new_records, visited_states = self._filter_states(records, new_records, visited_states)
            records, visited_states = self._resample_records(
                resampler, new_records, visited_states, terminal_indices, step, width
            )

            if not records:
                break

        logger.info("")

        return [record.state for record in records] if records else [state]
