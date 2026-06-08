from statistics import mean
from contextvars import ContextVar
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
        self.width_confidence_threshold = float(getattr(config, "width_confidence_threshold", 0.70))
        self.width_weak_threshold = float(getattr(config, "width_weak_threshold", 0.50))
        self.width_uncertainty_threshold = float(getattr(config, "width_uncertainty_threshold", 0.25))
        self._width_plateau_steps: ContextVar[int] = ContextVar("reagents_v3_width_plateau_steps", default=0)

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        token = self._width_plateau_steps.set(0)
        try:
            return await super().solve(idx=idx, state=state, namespace=namespace, value_cache=value_cache)
        finally:
            self._width_plateau_steps.reset(token)

    def _topk_mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        k = max(1, min(self.width_top_k, len(values)))
        return float(mean(sorted(values, reverse=True)[:k]))

    def _bounded_width(self, width: int) -> int:
        min_width = max(1, min(self.min_width, self.max_width_bound))
        max_width = max(min_width, self.max_width_bound)
        return max(min_width, min(max_width, int(width)))

    def _value_scale(self, values: list[float]) -> float:
        return max(1.0, float(self.max_value), *(abs(value) for value in values))

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
        value_scale = self._value_scale(new_values)
        normalized_best = best_new / value_scale
        normalized_topk = topk_new / value_scale
        normalized_spread = spread_new / value_scale
        terminal_count = sum(1 for record in new_records if self.env.is_final(record.state))

        old_width = width
        decision = "keep"
        plateau_steps = self._width_plateau_steps.get()

        if terminal_count == len(new_records):
            plateau_steps = 0
            width = self.min_width
            decision = "shrink_all_terminal"
        elif normalized_spread >= self.width_uncertainty_threshold:
            plateau_steps = 0
            decision = "keep_uncertain"
        elif normalized_best >= self.width_confidence_threshold:
            plateau_steps = 0
            width -= 1
            decision = "shrink_confident"
        elif normalized_topk < self.width_weak_threshold:
            plateau_steps += 1
            if plateau_steps >= self.width_patience:
                width += 1
                decision = "grow_weak_plateau"
            else:
                decision = "keep_weak_waiting"
        else:
            plateau_steps = 0
            decision = "keep_moderate"

        self._width_plateau_steps.set(plateau_steps)

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
            "value_scale": value_scale,
            "normalized_best": normalized_best,
            "normalized_topk": normalized_topk,
            "normalized_spread": normalized_spread,
            "terminal_count": terminal_count,
            "plateau_steps": plateau_steps,
            "decision": decision,
            "min_width": self.min_width,
            "max_width": self.max_width_bound,
            "confidence_threshold": self.width_confidence_threshold,
            "weak_threshold": self.width_weak_threshold,
            "uncertainty_threshold": self.width_uncertainty_threshold,
        })
        return new_width
