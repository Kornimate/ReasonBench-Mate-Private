import random
import asyncio
from typing import TypedDict
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
import logging
from ..logging_utils import action_summary, log_event, log_section, log_section_end, score_summary, terminal_summary

logger = logging.getLogger("__main__")

@AgentDictFactory.register
class AgentDictTOT(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    evaluate_params: DecodingParameters

@MethodFactory.register
class MethodTOT_DFS(Method):
    def __init__(self,
                model: Model,
                agents: AgentDictTOT,
                env: Environment,
                config: OmegaConf
                ):
        super().__init__(model, agents, env, config)
        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.evaluate_params = agents["evaluate_params"]

        self.num_selections = config.num_selections
        self.num_steps = config.num_steps
        self.num_evaluations = config.num_evaluations

        self.pruning_threshold = config.get("pruning_threshold", None)
        self.max_iterations = config.get("max_iterations", None)

        """
        max_iterations: Attempts Unique branches
        """

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):

        ## cachesaver
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]

        output = []
        iteration_count = 0
        log_section("ToT-DFS Method Information:")

        # Inner recursive DFS function
        async def dfs(s, t):
            nonlocal iteration_count

            if t >= self.num_steps:
                iteration_count += 1
                
                action_coroutines = [

                    self.step_agent.act(
                        model=self.model,
                        state=state,
                        namespace=namespace,
                        request_id=f"idx{idx}-step{t}-{hash(state)}-agent{i}",
                        params=self.step_params,
                    )
                    for i, state in enumerate(s)
                ]

                actions = await asyncio.gather(*action_coroutines)

                state_proposals = []
                for state2, state_actions in zip(s, actions):
                    for action in state_actions:
                        next_state = self.env.step(state2, action)
                        state_proposals.append(next_state)
                
                if state_proposals == []:
                    log_event("TOT_DFS_LEAF", {
                        "idx": idx,
                        "depth": t,
                        "iteration": iteration_count,
                        "frontier_size": len(s),
                        "proposal_count": 0,
                        "actions": action_summary(actions),
                        "empty_proposals": True,
                    })
                    return False

                value_coroutines = [
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{t}-{hash(state)}-agent{i}",
                        params=self.evaluate_params,
                        cache=value_cache
                    )
                    for i, state in enumerate(state_proposals)
                ]

                values = await asyncio.gather(*value_coroutines)
                state_value_pairs = list(zip(state_proposals, values))

                if not state_value_pairs:
                    return False
                
                best_state, best_value = max(state_value_pairs, key=lambda x: x[1])
                output.append((best_state, best_value))
                solved = self.env.evaluate(best_state)[1] == 1
                log_event("TOT_DFS_LEAF", {
                    "idx": idx,
                    "depth": t,
                    "iteration": iteration_count,
                    "frontier_size": len(s),
                    "proposal_count": len(state_proposals),
                    "actions": action_summary(actions),
                    "values": score_summary(values),
                    "best_value": best_value,
                    "terminal": terminal_summary(self.env, [best_state]),
                    "solved": solved,
                })

                if solved: ## early stopping
                    return True  # returning a list with best_state for consistency

                if (self.max_iterations is not None and iteration_count >= self.max_iterations):
                    print(f"Early stopping: Max iterations reached")
                    log_event("TOT_DFS_STOP", {
                        "idx": idx,
                        "reason": "max_iterations",
                        "iteration_count": iteration_count,
                        "max_iterations": self.max_iterations,
                    })
                    return True
                return False
                



            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{t}-{hash(state)}-agent{i}",
                    params=self.step_params,
                )
                for i, state in enumerate(s)
            ]
            actions = await asyncio.gather(*action_coroutines)

            state_proposals = []
            for state2, state_actions in zip(s, actions):
                for action in state_actions:
                    next_state = self.env.step(state2, action)
                    state_proposals.append(next_state)
            
            if state_proposals == []:
                log_event("TOT_DFS_STEP", {
                    "idx": idx,
                    "depth": t,
                    "frontier_size": len(s),
                    "proposal_count": 0,
                    "actions": action_summary(actions),
                    "empty_proposals": True,
                })
                return False


            # Evaluate all proposals
            value_coroutines = [
                self.eval_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_evaluations,
                    namespace=namespace,
                    request_id=f"idx{idx}-evaluation{t}-{hash(state)}-agent{i}",
                    params=self.evaluate_params,
                    cache=value_cache
                )
                for i, state in enumerate(state_proposals)
            ]
            values = await asyncio.gather(*value_coroutines)

            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            selected_pairs = [
                (state2, value)
                for state2, value in sorted_pairs[:self.num_selections]
                if self.pruning_threshold is None or value > self.pruning_threshold
            ]
            log_event("TOT_DFS_STEP", {
                "idx": idx,
                "depth": t,
                "frontier_size": len(s),
                "proposal_count": len(state_proposals),
                "selected_count": len(selected_pairs),
                "pruned_count": max(0, min(len(sorted_pairs), self.num_selections) - len(selected_pairs)),
                "actions": action_summary(actions),
                "values": score_summary(values),
                "pruning_threshold": self.pruning_threshold,
            })

            for state2, value in selected_pairs:
                if await dfs([state2], t + 1): # Go one step deeper in the DFS search
                    return True
            return False
        
        try:
            result = await dfs(states, 1)
        except Exception as e:
            print(f"Error during initial dfs call: {e}")
            log_event("TOT_DFS_ERROR", {
                "idx": idx,
                "error": str(e),
            })

        output = sorted(output, key=lambda x: x[1], reverse=True)[:self.num_selections] if output else [states]
        log_event("TOT_DFS_RESULT", {
            "idx": idx,
            "iteration_count": iteration_count,
            "output_count": len(output),
        })
        log_section_end()
        return [x[0] for x in output]
        
        
