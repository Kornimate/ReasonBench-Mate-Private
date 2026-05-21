import random
import asyncio
import logging
from typing import TypedDict
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
from ..logging_utils import action_summary, log_event, log_section, log_section_end, score_summary, terminal_summary
logger = logging.getLogger(__name__)

@AgentDictFactory.register
class AgentDictGOT(TypedDict):
    step: Agent
    aggregate: Agent
    evaluate: Agent
    step_params: DecodingParameters
    aggregate_params: DecodingParameters
    evaluate_params: DecodingParameters

@MethodFactory.register
class MethodGOT(Method):
    def __init__(self, 
                 model: Model, 
                 agents: AgentDictGOT, 
                 env: Environment,
                 config: OmegaConf
                 ):
        super().__init__(model, agents, env, config)

        self.step_agent = agents["step"]
        self.aggregate_agent = agents["aggregate"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.aggregate_params = agents["aggregate_params"]
        self.evaluate_params = agents["evaluate_params"]

        self.num_selections = config.num_selections
        self.num_steps = config.num_steps
        self.num_generate = config.num_generate
        self.num_best = config.num_best
        self.num_evaluations = config.num_evaluations

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]
        logger.debug(f"Solving game: {idx}")

        solved = False
        log_section("GoT Method Information:")
        for step in range(self.num_steps):
            if solved:
                logger.debug(f"Task {idx} solved at step {step - 1}.")
                break

            logger.debug(f"Step: {step} ({idx})")
            # Generate actions for each state
            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_generate,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
                    params=self.step_params,
                )
                for i, state in enumerate(states)
            ]
            generated_actions = await asyncio.gather(*action_coroutines)
            logger.debug(f"{len(generated_actions)} Actions generated for task {idx}; \n {generated_actions}")

            # Aggregate actions
            aggregate_coroutines = [
                self.aggregate_agent.act(
                    model=self.model,
                    state=state,
                    actions=action,
                    k=self.num_selections,
                    namespace=namespace,
                    request_id=f"idx{idx}-aggregate{step}-{hash(state)}-agent{i}",
                    params=self.aggregate_params,
                )
                for i, (state, action) in enumerate(zip(states, generated_actions))
            ]

            selected_actions = await asyncio.gather(*aggregate_coroutines)
            logger.debug(f"{len(selected_actions)} Actions selected for task {idx}: \n{selected_actions}")

            # Execute actions on environment
            proposed_states = []
            for state, state_actions in zip(states, selected_actions):
                for action in state_actions:
                    proposed_states.append(self.env.step(state, action))
            
            if proposed_states == []:
                log_event("GOT_STEP", {
                    "idx": idx,
                    "step": step,
                    "frontier_size": len(states),
                    "generated": action_summary(generated_actions),
                    "aggregated": action_summary(selected_actions),
                    "proposal_count": 0,
                    "empty_proposals": True,
                })
                break
            
            # Early stop in case any state is solved
            if any(self.env.evaluate(state)[1] == 1 for state in states):
                solved = True
            
            logger.debug(f"Env step for task {idx}: \n{proposed_states}")
            # Evaluate all proposals
            value_coroutines = [
                self.eval_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_evaluations,
                    namespace=namespace,
                    request_id=f"idx{idx}-evaluation{step}-{hash(state)}-agent{i}",
                    params=self.evaluate_params,
                    cache=value_cache
                )
                for i, state in enumerate(proposed_states)
            ]
            values = await asyncio.gather(*value_coroutines)
            logger.debug(f"Values given for task {idx}: \n{values}")

            # Choose the best states based on their value
            state_value_pairs = list(zip(proposed_states, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values = map(list, zip(*sorted_pairs[:self.num_best]))
            solved = any(self.env.evaluate(state)[1] == 1 for state in states)
            log_event("GOT_STEP", {
                "idx": idx,
                "step": step,
                "frontier_size": len(states),
                "generated": action_summary(generated_actions),
                "aggregated": action_summary(selected_actions),
                "proposal_count": len(proposed_states),
                "selected_count": len(states),
                "values": score_summary(values),
                "terminal": terminal_summary(self.env, states),
                "solved": solved,
            })
        
        log_section_end()
        return states
