import asyncio
import random
import math
import logging
import json
from dataclasses import dataclass
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
class FeatureBasedFleetSelector:
    num_steps: int
    min_react_prob: float = 0.15
    max_react_prob: float = 0.85
    min_count_per_type: int = 1

    beta_0: float = -0.4
    beta_difficulty: float = 1.2
    beta_uncertainty: float = 1.0
    beta_stagnation: float = 0.8
    beta_duplicate: float = 0.8
    beta_progress: float = -0.8
    beta_time_pressure: float = -0.7

    previous_best_value: float = 0.0
    stagnation_steps: int = 0

    def sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def compute_features(
        self,
        records,
        step: int,
        difficulty: float = 0.5,
    ) -> dict:
        values = np.array([float(record.value) for record in records], dtype=float)
        width = len(records)

        mean_value = float(np.mean(values)) if width else 0.0
        best_value = float(np.max(values)) if width else 0.0
        value_variance = float(np.var(values)) if width else 0.0

        # Assumes evaluator values are in [0, 1].
        uncertainty = min(1.0, value_variance / 0.25)

        progress = best_value - self.previous_best_value
        progress_norm = max(0.0, min(1.0, progress))

        if progress > 1e-6:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1

        stagnation_norm = min(1.0, self.stagnation_steps / 3.0)

        state_strings = [str(record.state) for record in records]
        unique_states = len(set(state_strings))
        duplicate_fraction = 1.0 - (unique_states / max(width, 1))

        time_pressure = step / max(self.num_steps - 1, 1)

        self.previous_best_value = max(self.previous_best_value, best_value)

        return {
            "width": width,
            "mean_value": mean_value,
            "best_value": best_value,
            "value_variance": value_variance,
            "uncertainty": uncertainty,
            "progress": progress,
            "progress_norm": progress_norm,
            "stagnation_norm": stagnation_norm,
            "duplicate_fraction": duplicate_fraction,
            "difficulty": max(0.0, min(1.0, difficulty)),
            "time_pressure": max(0.0, min(1.0, time_pressure)),
        }

    def react_probability(self, features: dict) -> float:
        z = (
            self.beta_0
            + self.beta_difficulty * features["difficulty"]
            + self.beta_uncertainty * features["uncertainty"]
            + self.beta_stagnation * features["stagnation_norm"]
            + self.beta_duplicate * features["duplicate_fraction"]
            + self.beta_progress * features["progress_norm"]
            + self.beta_time_pressure * features["time_pressure"]
        )

        p_react = self.sigmoid(z)

        p_react = max(self.min_react_prob, min(self.max_react_prob, p_react))

        return p_react

    def allocate(
        self,
        records,
        step: int,
        difficulty: float = 0.5,
    ) -> tuple[list[int], dict]:
        features = self.compute_features(
            records=records,
            step=step,
            difficulty=difficulty,
        )

        width = features["width"]
        p_react = self.react_probability(features)
        p_act = 1.0 - p_react

        counts = self._largest_remainder_allocation(
            width=width,
            probs=np.array([p_act, p_react], dtype=float),
        )

        counts = self._enforce_min_counts(counts, width)

        features["p_act"] = p_act
        features["p_react"] = p_react
        features["n_act"] = int(counts[0])
        features["n_react"] = int(counts[1])

        return counts.tolist(), features

    def _largest_remainder_allocation(
        self,
        width: int,
        probs: np.ndarray,
    ) -> np.ndarray:
        raw = width * probs
        counts = np.floor(raw).astype(int)

        remaining = width - int(counts.sum())
        fractional = raw - counts
        order = np.argsort(-fractional)

        for idx in order[:remaining]:
            counts[idx] += 1

        return counts

    def _enforce_min_counts(
        self,
        counts: np.ndarray,
        width: int,
    ) -> np.ndarray:
        if self.min_count_per_type <= 0:
            return counts

        required_total = 2 * self.min_count_per_type
        if width < required_total:
            return counts

        for i in range(2):
            if counts[i] < self.min_count_per_type:
                donor = int(np.argmax(counts))
                if counts[donor] > self.min_count_per_type:
                    counts[donor] -= 1
                    counts[i] += 1

        return counts

@AgentDictFactory.register
class AgentDictReagentsFAS(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentInfo]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagentsFAS(Method):
    def __init__(
        self,
        agents: AgentDictReagentsFAS,
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

        self.selector = FeatureBasedFleetSelector(
            num_steps=self.num_steps,
            min_react_prob=float(getattr(config, "min_react_prob", 0.15)),
            max_react_prob=float(getattr(config, "max_react_prob", 0.85)),
            min_count_per_type=int(getattr(config, "min_count_per_type", 1)),
            beta_0=float(getattr(config, "beta_0", -0.4)),
            beta_difficulty=float(getattr(config, "beta_difficulty", 1.2)),
            beta_uncertainty=float(getattr(config, "beta_uncertainty", 1.0)),
            beta_stagnation=float(getattr(config, "beta_stagnation", 0.8)),
            beta_duplicate=float(getattr(config, "beta_duplicate", 0.8)),
            beta_progress=float(getattr(config, "beta_progress", -0.8)),
            beta_time_pressure=float(getattr(config, "beta_time_pressure", -0.7)),
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
        difficulty: float = 0.5,
    ):
        fleet_counts, selector_features = self.selector.allocate(
            records=records,
            step=step,
            difficulty=difficulty,
        )

        width = len(records)

        agent_indices = []
        for agent_index, count in enumerate(fleet_counts):
            agent_indices.extend([agent_index] * count)

        agent_indices = agent_indices[:width]

        while len(agent_indices) < width:
            # fallback: use the larger count type
            agent_indices.append(int(np.argmax(fleet_counts)))

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

        return new_records, agent_indices, fleet_counts, selector_features

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

        logger.info("Runtime Agent Distribution Information:")

        for step in range(self.num_steps):
            difficulty = getattr(self, "current_difficulty", 0.5)

            new_records, agent_indices, fleet_counts, selector_features = await self._mutate_states(
                records=records,
                namespace=namespace,
                idx=idx,
                step=step,
                difficulty=difficulty,
            )
            
            logger.info(
                '\t' + json.dumps({
                    "step": step,
                    "width": len(records),
                    "fleet_counts": fleet_counts,
                    "selector_features": selector_features,
                })
            )

            new_records, terminal_indices, solved_indices = await self._evaluate_states(
                new_records,
                value_cache,
                namespace,
                idx,
                step,
            )

            self.priors = np.nan_to_num(self.priors, nan=0.0)
            self.priors /= self.priors.sum(axis=-1, keepdims=True)
            self._update_priors(agent_indices, records, new_records)
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
