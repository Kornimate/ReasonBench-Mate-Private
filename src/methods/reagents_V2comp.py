import math
from typing import Optional, TypedDict

import numpy as np
from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from ..logging_utils import log_event
from .reagents import DifficultyAgentSpec, MethodReagents, StepAgentSpec


@AgentDictFactory.register
class AgentDictReagents_V2comp(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_V2comp(MethodReagents):
    def __init__(
        self,
        agents: AgentDictReagents_V2comp,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)

        self.comparison_temperature = float(getattr(config, "comparison_temperature", 0.25))
        self.comparison_epsilon = float(getattr(config, "comparison_epsilon", 0.05))

    def _comparison_probs(self, depth: int) -> np.ndarray:
        bounded_depth = max(0, min(depth, self.num_steps - 1))
        priors = np.nan_to_num(self.priors[bounded_depth], nan=0.0)
        total = priors.sum()
        if total <= 0:
            priors = np.ones(len(self.step_agents), dtype=float) / len(self.step_agents)
        else:
            priors = priors / total

        temperature = max(self.comparison_temperature, 1e-6)
        comparative_scores = np.zeros(len(self.step_agents), dtype=float)
        for i in range(len(self.step_agents)):
            for j in range(len(self.step_agents)):
                if i == j:
                    continue
                advantage = (priors[i] - priors[j]) / temperature
                comparative_scores[i] += 1.0 / (1.0 + math.exp(-advantage))

        centered = comparative_scores - np.max(comparative_scores)
        exp_scores = np.exp(centered)
        probs = exp_scores / exp_scores.sum()
        epsilon = max(0.0, min(1.0, self.comparison_epsilon))
        probs = (1.0 - epsilon) * probs + epsilon / len(probs)
        return probs / probs.sum()

    def _sample_agent_index(self, depth: int) -> int:
        probs = self._comparison_probs(depth)
        selected = int(np.random.choice(len(self.step_agents), p=probs))
        log_event("REAGENTS_V2COMP_SELECTION", {
            "depth": depth,
            "selected_agent_index": selected,
            "comparison_probs": probs.tolist(),
        })
        return selected
