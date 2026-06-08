from statistics import mean
from typing import Optional, TypedDict

from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..logging_utils import log_event
from ..typedefs import Agent, DecodingParameters, Environment, Model, State
from .reagents import DifficultyAgentSpec, MethodReagents, SearchRecord, StepAgentSpec


@AgentDictFactory.register
class AgentDictReagents_v3(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v3(MethodReagents):
    def __init__(
        self,
        agents: AgentDictReagents_v3,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.min_width = int(getattr(config, "min_width", max(1, self.width // 2)))
        self.max_width_bound = int(getattr(config, "max_width", self.width + self.width // 2))
        self.width_top_k = int(getattr(config, "width_top_k", 2))
        self.width_patience = int(getattr(config, "width_patience", 2))
        self.width_improvement_threshold = float(getattr(config, "width_improvement_threshold", 0.05))
        self.width_uncertainty_threshold = float(getattr(config, "width_uncertainty_threshold", 0.25))
        self.width_plateau_steps = 0

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        self.width_plateau_steps = 0
        return await super().solve(idx=idx, state=state, namespace=namespace, value_cache=value_cache)

    def _topk_mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        k = max(1, min(self.width_top_k, len(values)))
        return float(mean(sorted(values, reverse=True)[:k]))

    def _bounded_width(self, width: int) -> int:
        min_width = max(1, min(self.min_width, self.max_width_bound))
        max_width = max(min_width, self.max_width_bound)
        return max(min_width, min(max_width, int(width)))

    def _update_width(self, old_records: list[SearchRecord], new_records: list[SearchRecord], width: int) -> int:
        if not self.features["runtime_width_adaptation"] or not old_records or not new_records:
            return width

        old_values = [float(record.value) for record in old_records]
        new_values = [float(record.value) for record in new_records]

        best_old = max(old_values)
        best_new = max(new_values)
        topk_old = self._topk_mean(old_values)
        topk_new = self._topk_mean(new_values)
        spread_new = best_new - min(new_values)
        progress = best_new - best_old
        topk_progress = topk_new - topk_old

        old_width = width
        decision = "keep"

        if best_new >= self.max_value:
            self.width_plateau_steps = 0
            width = self.min_width
            decision = "shrink_solved"
        elif (
            progress >= self.width_improvement_threshold
            or topk_progress >= self.width_improvement_threshold
        ):
            self.width_plateau_steps = 0
            if spread_new >= self.width_uncertainty_threshold:
                decision = "keep_improved_uncertain"
            else:
                width -= 1
                decision = "shrink_improved_confident"
        else:
            self.width_plateau_steps += 1
            if self.width_plateau_steps >= self.width_patience:
                width += 1
                decision = "grow_plateau"
            else:
                decision = "keep_waiting"

        new_width = self._bounded_width(width)
        log_event("REAGENTS_V3_WIDTH", {
            "old_width": old_width,
            "new_width": new_width,
            "best_old": best_old,
            "best_new": best_new,
            "topk_old": topk_old,
            "topk_new": topk_new,
            "spread_new": spread_new,
            "progress": progress,
            "topk_progress": topk_progress,
            "plateau_steps": self.width_plateau_steps,
            "decision": decision,
            "min_width": self.min_width,
            "max_width": self.max_width_bound,
        })
        return new_width
