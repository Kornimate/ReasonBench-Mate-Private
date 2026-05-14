import asyncio
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
from .reagents_aos import StepAgentInfo

logger = logging.getLogger("__main__")

@dataclass
class ALNSFleetAllocator:
    num_agent_types: int
    reaction_factor: float = 0.2
    min_weight: float = 0.05
    min_count_per_type: int = 1

    weights: np.ndarray = field(init=False)
    segment_scores: np.ndarray = field(init=False)
    segment_counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.weights = np.ones(self.num_agent_types, dtype=float)
        self.segment_scores = np.zeros(self.num_agent_types, dtype=float)
        self.segment_counts = np.zeros(self.num_agent_types, dtype=int)

    def get_probs(self) -> np.ndarray:
        safe_weights = np.maximum(self.weights, self.min_weight)
        return safe_weights / safe_weights.sum()

    def allocate(self, width: int) -> list[int]:
        probs = self.get_probs()
        raw = width * probs

        counts = np.floor(raw).astype(int)
        remaining = width - int(counts.sum())

        fractional = raw - counts
        order = np.argsort(-fractional)

        for idx in order[:remaining]:
            counts[idx] += 1

        counts = self._enforce_min_counts(counts, width)

        return counts.tolist()

    def _enforce_min_counts(self, counts: np.ndarray, width: int) -> np.ndarray:
        if self.min_count_per_type <= 0:
            return counts

        required_total = self.num_agent_types * self.min_count_per_type
        if width < required_total:
            return counts

        for i in range(self.num_agent_types):
            if counts[i] < self.min_count_per_type:
                deficit = self.min_count_per_type - counts[i]

                for _ in range(deficit):
                    donor = int(np.argmax(counts))
                    if counts[donor] <= self.min_count_per_type:
                        break

                    counts[donor] -= 1
                    counts[i] += 1

        return counts

    def add_result(self, agent_index: int, score: float) -> None:
        self.segment_scores[agent_index] += float(score)
        self.segment_counts[agent_index] += 1

    def update_weights(self) -> None:
        for i in range(self.num_agent_types):
            if self.segment_counts[i] == 0:
                continue

            avg_score = self.segment_scores[i] / self.segment_counts[i]

            self.weights[i] = (
                (1.0 - self.reaction_factor) * self.weights[i]
                + self.reaction_factor * avg_score
            )

            self.weights[i] = max(self.weights[i], self.min_weight)

        self.segment_scores[:] = 0.0
        self.segment_counts[:] = 0

@AgentDictFactory.register
class AgentDictReagentsALNS(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentInfo]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagentsALNS(Method):
    def __init__(
        self,
        agents: AgentDictReagentsALNS,
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
        
        self.alns_score_solved = float(getattr(config, "alns_score_solved", 10.0))
        self.alns_score_new_best = float(getattr(config, "alns_score_new_best", 5.0))
        self.alns_score_improved = float(getattr(config, "alns_score_improved", 2.0))
        self.alns_score_neutral = float(getattr(config, "alns_score_neutral", 0.5))
        self.alns_score_worse = float(getattr(config, "alns_score_worse", 0.0))

        self.features = {
            "updating_priors": bool(getattr(config, "updating_priors", True)),
            "difficulty_based_width_init": bool(getattr(config, "difficulty_based_width_init", True)),
            "runtime_width_adaptation": bool(getattr(config, "runtime_width_adaptation", True)),
            "skewed_state_detection": bool(getattr(config, "skewed_state_detection", False)),
        }

        self.priors = np.ones((self.num_steps, len(self.step_agents))) / max(len(self.step_agents), 1)
        
        self.allocator = ALNSFleetAllocator(
            num_agent_types=len(self.step_agents),
            reaction_factor=(getattr(config, "alns_reaction_factor", 0.2)),
            min_weight=(getattr(config, "alns_min_weight", 0.05)),
            min_count_per_type=int(getattr(config, "alns_min_count_per_type", 1)),
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

        fleet_counts = self.allocator.allocate(width)

        agent_indices = []
        for agent_index, count in enumerate(fleet_counts):
            agent_indices.extend([agent_index] * count)

        agent_indices = agent_indices[:width]

        while len(agent_indices) < width:
            best_agent = int(np.argmax(self.allocator.get_probs()))
            agent_indices.append(best_agent)

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

        return new_records, agent_indices, fleet_counts

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

    def compute_alns_score(
        self,
        old_value: float,
        new_value: float,
        best_value_before_step: float,
        solved: bool = False,
        terminal: bool = False,
    ) -> float:
        if solved:
            return self.alns_score_solved

        if terminal and new_value < old_value:
            return self.alns_score_worse

        if new_value > best_value_before_step:
            return self.alns_score_new_best

        if new_value > old_value:
            return self.alns_score_improved

        if new_value == old_value:
            return self.alns_score_neutral

        return 0.0 # could be self.alns_score_worse or a separate score for worse outcomes

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

        for step in range(self.num_steps):
            best_value_before_step = max(record.value for record in records)

            new_records, agent_indices, fleet_counts = await self._mutate_states(
                records,
                namespace,
                idx,
                step,
            )
            logger.info(
                {
                    "step": step,
                    "width": len(records),
                    "fleet_counts": fleet_counts,
                    "weights": self.allocator.weights.tolist(),
                    "probs": self.allocator.get_probs().tolist(),
                    "num_act": fleet_counts[0],
                    "num_react": fleet_counts[1],
                }
            )

            new_records, terminal_indices, solved_indices = await self._evaluate_states(
                new_records,
                value_cache,
                namespace,
                idx,
                step,
            )

            terminal_set = set(terminal_indices)
            solved_set = set(solved_indices)

            for i, (old_record, new_record, agent_index) in enumerate(
                zip(records, new_records, agent_indices)
            ):
                score = self.compute_alns_score(
                    old_value=old_record.value,
                    new_value=new_record.value,
                    best_value_before_step=best_value_before_step,
                    solved=i in solved_set,
                    terminal=i in terminal_set,
                )

                self.allocator.add_result(agent_index, score)

            logger.info(
                {
                    "step": step,
                    "segment_scores": self.allocator.segment_scores.tolist(),
                    "segment_counts": self.allocator.segment_counts.tolist(),
                }
            )
            self.allocator.update_weights()
            
            width = self._update_width(records, new_records, width)

            if solved_indices:
                return [new_records[i].state for i in solved_indices]

            new_records, visited_states = self._filter_states(records, new_records, visited_states)
            records, visited_states = self._resample_records(
                resampler, new_records, visited_states, terminal_indices, step, width
            )

            if not records:
                break

        return [record.state for record in records] if records else [state]
