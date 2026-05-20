import asyncio
import random
from dataclasses import dataclass
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, MAX_SEED, Method, Model, State
from ..utils import Resampler
from .logging_utils import log_event, log_section, log_section_end, score_summary, terminal_summary


class StepAgentSpec(TypedDict):
    agent: Agent
    params: DecodingParameters
    num_agents: int


class DifficultyAgentSpec(TypedDict):
    agent: Agent
    params: DecodingParameters


@dataclass(frozen=True)
class SearchRecord:
    state: State
    value: float
    depth: int
    uncertainty: float = 0.0


@AgentDictFactory.register
class AgentDictReagents(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents(Method):
    def __init__(
        self,
        agents: AgentDictReagents,
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
        agent_indices = []
        coroutines = []

        for i, record in enumerate(records):
            agent_index = self._sample_agent_index(record.depth)
            spec = self.step_agents[agent_index]
            agent_indices.append(agent_index)
            coroutines.append(
                spec["agent"].act(
                    model=self.model,
                    state=record.state,
                    n=1,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(record.state)}-agent{agent_index + 100 * i}",
                    params=spec["params"],
                )
            )

        action_batches = await asyncio.gather(*coroutines)
        new_records = []
        for record, actions in zip(records, action_batches):
            if not actions:
                new_records.append(SearchRecord(state=record.state, value=record.value, depth=record.depth + 1))
                continue
            try:
                new_state = self.env.step(record.state, actions[0])
            except Exception:
                new_state = record.state
            new_records.append(SearchRecord(state=new_state, value=record.value, depth=record.depth + 1))

        return new_records, agent_indices

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

    def _update_priors(self, agent_indices: list[int], old_records: list[SearchRecord], new_records: list[SearchRecord]) -> None:
        if not self.features["updating_priors"]:
            return

        for i, agent_index in enumerate(agent_indices):
            depth = max(0, min(old_records[i].depth, self.num_steps - 1))
            delta = new_records[i].value - old_records[i].value
            for offset in range(-self.k, self.k + 1):
                target_depth = depth + offset
                if 0 <= target_depth < self.num_steps:
                    self.priors[target_depth][agent_index] += (
                        (self.backtrack ** abs(offset)) * self.priors[depth][agent_index] * self.alpha * delta
                    )
                    self.priors[target_depth][agent_index] = min(1.0, max(0.001, self.priors[target_depth][agent_index]))

        self.priors = np.nan_to_num(self.priors, nan=0.0)
        self.priors /= self.priors.sum(axis=-1, keepdims=True)

    def _update_width(self, old_records: list[SearchRecord], new_records: list[SearchRecord], width: int) -> int:
        if not self.features["runtime_width_adaptation"] or not old_records or not new_records:
            return width

        old_avg = sum(record.value for record in old_records) / len(old_records)
        new_avg = sum(record.value for record in new_records) / len(new_records)

        if old_avg < new_avg:
            return max(max(1, self.width // 2), width - 1)
        return min(self.width + self.width // 2, width + 1)

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

        log_section("ReAgents Method Information:")
        for step in range(self.num_steps):
            new_records, agent_indices = await self._mutate_states(records, namespace, idx, step)
            new_records, terminal_indices, solved_indices = await self._evaluate_states(
                new_records, value_cache, namespace, idx, step
            )

            self.priors = np.nan_to_num(self.priors, nan=0.0)
            self.priors /= self.priors.sum(axis=-1, keepdims=True)
            old_width = width
            old_values = [record.value for record in records]
            new_values = [record.value for record in new_records]
            self._update_priors(agent_indices, records, new_records)
            width = self._update_width(records, new_records, width)
            agent_types = [
                self.step_agents[agent_index].get("agent_type", f"agent_{agent_index}")
                for agent_index in agent_indices
            ]
            log_event("REAGENTS_STEP", {
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
                "old_values": score_summary(old_values),
                "new_values": score_summary(new_values),
                "terminal_indices": terminal_indices,
                "solved_indices": solved_indices,
                "priors": self.priors.tolist(),
                "terminal": terminal_summary(self.env, [record.state for record in new_records]),
            })

            if solved_indices:
                log_section_end()
                return [new_records[i].state for i in solved_indices]

            new_records, visited_states = self._filter_states(records, new_records, visited_states)
            records, visited_states = self._resample_records(
                resampler, new_records, visited_states, terminal_indices, step, width
            )

            if not records:
                break

        log_section_end()
        return [record.state for record in records] if records else [state]
