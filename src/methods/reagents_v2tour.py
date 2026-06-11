from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from ..logging_utils import log_event
from .reagents import DifficultyAgentSpec, MethodReagents, StepAgentSpec


@AgentDictFactory.register
class AgentDictReagents_v2tour(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v2tour(MethodReagents):
    def __init__(
        self,
        agents: AgentDictReagents_v2tour,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.tournament_size = int(getattr(config, "tournament_size", 2))
        self.tournament_epsilon = float(getattr(config, "tournament_epsilon", 0.0))

    def _base_probs(self, depth: int) -> np.ndarray:
        bounded_depth = max(0, min(depth, self.num_steps - 1))
        priors = np.nan_to_num(self.priors[bounded_depth], nan=0.0)
        total = priors.sum()
        if total <= 0:
            return np.ones(len(self.step_agents), dtype=float) / len(self.step_agents)
        probs = priors / total
        probs = np.maximum(probs, 1e-9)
        return probs / probs.sum()

    def _sample_agent_index(self, depth: int) -> int:
        probs = self._base_probs(depth)
        epsilon = max(0.0, min(1.0, self.tournament_epsilon))
        if np.random.random() < epsilon:
            selected = int(np.random.choice(len(self.step_agents)))
            candidates = [selected]
        else:
            tournament_size = max(1, min(self.tournament_size, len(self.step_agents)))
            candidates = np.random.choice(
                len(self.step_agents),
                size=tournament_size,
                replace=False,
            ).tolist()
            selected = max(candidates, key=lambda agent_index: probs[agent_index])

        log_event("REAGENTS_V2TOUR_SELECTION", {
            "depth": depth,
            "selected_agent_index": selected,
            "candidate_agent_indices": candidates,
            "prior_probs": probs.tolist(),
        })
        return int(selected)
