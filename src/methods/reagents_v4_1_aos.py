from typing import Optional, TypedDict

from omegaconf import OmegaConf

from .. import AgentDictFactory, MethodFactory
from ..typedefs import Agent, DecodingParameters, Environment, Model
from .reagents import DifficultyAgentSpec, StepAgentSpec
from .reagents_v3_aos import MethodReagents_v3_aos


@AgentDictFactory.register
class AgentDictReagents_v4_1_aos(TypedDict):
    evaluate: Agent
    evaluate_params: DecodingParameters
    step_agents: list[StepAgentSpec]
    difficulty_agent: Optional[DifficultyAgentSpec]


@MethodFactory.register
class MethodReagents_v4_1_aos(MethodReagents_v3_aos):
    """ReAgents v4 AOS with one added heterogeneous role."""

    def __init__(
        self,
        agents: AgentDictReagents_v4_1_aos,
        model: Model,
        env: Environment,
        config: OmegaConf,
    ):
        super().__init__(agents=agents, model=model, env=env, config=config)
