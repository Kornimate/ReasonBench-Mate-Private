from typing import Optional, TypedDict

from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from .reagents import DifficultyAgentSpec, StepAgentSpec
from .reagents_v3_hierarchical import MethodReagents_v3_hierarchical


@AgentDictFactory.register
class AgentDictReagents_v4_2_hierarchical(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v4_2_hierarchical(MethodReagents_v3_hierarchical):
    """ReAgents v4 hierarchical search with two added heterogeneous roles."""

    def __init__(
        self,
        agents: AgentDictReagents_v4_2_hierarchical,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)
