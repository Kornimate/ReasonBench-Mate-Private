import random
import logging
import asyncio
from typing import TypedDict, List, Dict, Any, Tuple
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
logger = logging.getLogger(__name__)

# THIS IS JUST EXPERIMENTAL AND NOT JUSTIFIED

@AgentDictFactory.register
class AgentDictReAgents(TypedDict):
    step: Agent # ActAgent
    evaluate: Agent # EvaluateAgent
    predict: Agent
    step_params: DecodingParameters
    evaluate_params: DecodingParameters
    predict_params: DecodingParameters

@MethodFactory.register
class MethodReAgents(Method):
    def __init__(self, 
                 agents: AgentDictReAgents,
                 model: Model, 
                 env: Environment, 
                 config: OmegaConf
                 ):
        # Old implementation:
        # self.step_agent = agents["step"]
        # self.eval_agent = agents["evaluate"]
        #
        # self.step_params = agents["step_params"]
        # self.evaluate_params = agents["evaluate_params"]
        #
        # self.num_agents = config.num_agents
        # self.num_steps = config.num_steps
        # self.k = config.k
        # self.backtrack = config.backtrack
        # self.resampling = config.resampling
        # self.origin = config.origin
        # self.min_steps = config.min_steps
        # self.num_evaluations = config.num_evaluations
        super().__init__(model, agents, env, config)

        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]
        self.predict_agent = agents["predict"]

        self.step_params = agents["step_params"]
        self.evaluate_params = agents["evaluate_params"]
        self.predict_params = agents["predict_params"]
        
        self.population_size = getattr(config, "population_size", 5)
        self.max_steps = getattr(config, "max_steps", 6)
        self.adaptive_scaling_enabled = getattr(config, "adaptive_scaling", True)
        self.heterogeneous_agents = getattr(config, "heterogeneous_agents", True)
        self.selection = getattr(config, "selection", "topk")
        self.num_evaluations = getattr(config, "num_evaluations", 1)

        self.low_score_threshold = getattr(config, "low_score_threshold", 0.3)
        self.high_score_threshold = getattr(config, "high_score_threshold", 0.8)
        self.extra_agents = getattr(config, "extra_agents", 2)
        self.min_population = getattr(config, "min_population", 1)
        self.tournament_size = getattr(config, "tournament_size", 3)
        self.heterogeneous_candidates = getattr(config, "heterogeneous_candidates", 4)


    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        # Old implementation:
        # randomness = idx
        # random.seed(randomness)
        # resampler = Resampler(randomness)
        #
        # visited_states = [("INIT", self.origin, state)]
        # initial_state = state
        # states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.num_agents)]
        #
        # solved = False
        # for step in range(self.num_steps):
        #     if solved:
        #         break
        #     ...
        #
        # if len(states) == 0:
        #     return [initial_state]
        # else:
        #     return states
        randomness = idx
        random.seed(randomness)
        target_population_size = await self.predict_initial_population(
            idx=idx,
            state=state,
            namespace=namespace,
        )

        population = await self.initialize_population(
            idx=idx,
            state=state,
            namespace=namespace,
            target_population_size=target_population_size,
        )

        if not population:
            return [state]

        for step in range(self.max_steps):
            population = await self.expand_population(
                idx=idx,
                population=population,
                step=step,
                namespace=namespace,
            )
            if not population:
                return [state]

            await self.evaluate_population(
                idx=idx,
                population=population,
                step=step,
                namespace=namespace,
                value_cache=value_cache,
            )

            solved_candidates = [
                agent for agent in population
                if self.env.evaluate(agent["state"])[1] == 1
            ]
            if solved_candidates:
                solved_candidates.sort(key=lambda agent: agent["score"], reverse=True)
                return [solved_candidates[0]["state"]]

            population = self.select_population(population, target_population_size)
            population, target_population_size = await self.adaptive_scale_population(
                idx=idx,
                population=population,
                step=step,
                namespace=namespace,
                target_population_size=target_population_size,
            )

        population.sort(key=lambda agent: agent["score"], reverse=True)
        return [population[0]["state"]] if population else [state]

    async def predict_initial_population(
        self,
        idx: int,
        state: State,
        namespace: str,
    ) -> int:
        # Old implementation: no equivalent method existed.
        # The initial size was fixed by:
        # states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.num_agents)]
        prediction = await self.predict_agent.act(
            model=self.model,
            state=state,
            max_agents=self.population_size,
            namespace=namespace,
            request_id=f"idx{idx}-predict-population",
            params=self.predict_params,
        )
        return max(self.min_population, min(self.population_size, int(prediction)))

    async def initialize_population(
        self,
        idx: int,
        state: State,
        namespace: str,
        target_population_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Initialize a population by sampling independent first-step continuations.

        Old implementation:
        - states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.num_agents)]
        """
        seeds = [
            state.clone(randomness=random.randint(0, MAX_SEED))
            for _ in range(target_population_size)
        ]
        initial_states = await self._advance_states(
            idx=idx,
            states=seeds,
            step=0,
            namespace=namespace,
            phase="init",
        )

        if not initial_states:
            initial_states = seeds

        return [{"state": agent_state, "score": 0.0} for agent_state in initial_states]

    async def expand_population(
        self,
        idx: int,
        population: List[Dict[str, Any]],
        step: int,
        namespace: str,
    ) -> List[Dict[str, Any]]:
        """
        Expand each population member by one step.

        Old implementation:
        - action_coroutines = [
        -     self.step_agent.act(
        -         model=self.model,
        -         state=state,
        -         n=1,
        -         namespace=namespace,
        -         request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
        -         params=self.step_params
        -     )
        -     for i, state in enumerate(states)
        - ]
        - actions = await asyncio.gather(*action_coroutines)
        - states = [self.env.step(state, action[0]) for state, action in zip(states, actions)]
        """
        active_states = []
        finished_agents = []

        for agent in population:
            if self.env.is_final(agent["state"]):
                finished_agents.append({"state": agent["state"], "score": agent["score"]})
            else:
                active_states.append(agent["state"])

        expanded_states = await self._advance_states(
            idx=idx,
            states=active_states,
            step=step,
            namespace=namespace,
            phase="expand",
        )

        return (
            [{"state": expanded_state, "score": 0.0} for expanded_state in expanded_states]
            + finished_agents
        )

    async def evaluate_population(
        self,
        idx: int,
        population: List[Dict[str, Any]],
        step: int,
        namespace: str,
        value_cache: dict = None,
    ) -> None:
        """
        Score all population members with the evaluation agent.

        Old implementation:
        - if step < self.num_steps-1 and self.k and step % self.k == 0:
        -     value_coroutines = [
        -         self.eval_agent.act(
        -             model=self.model,
        -             state=state,
        -             n=self.num_evaluations,
        -             namespace=namespace,
        -             request_id=f"idx{idx}-evaluation{step}-{hash(state)}-agent{i}",
        -             params=self.evaluate_params,
        -             cache=value_cache
        -         )
        -         for i, state in enumerate(states)
        -     ]
        -     values = await asyncio.gather(*value_coroutines)
        -     for i, (state, value) in enumerate(zip(states, values)):
        -         if i not in failed:
        -             visited_states.append((f"{i}.{step}", value, state))
        """
        value_coroutines = [
            self.eval_agent.act(
                model=self.model,
                state=agent["state"],
                n=self.num_evaluations,
                namespace=namespace,
                request_id=f"idx{idx}-evaluation{step}-{hash(agent['state'])}-agent{i}",
                params=self.evaluate_params,
                cache=value_cache,
            )
            for i, agent in enumerate(population)
        ]
        scores = await asyncio.gather(*value_coroutines)

        for agent, score in zip(population, scores):
            _, reward = self.env.evaluate(agent["state"])
            agent["score"] = max(self._normalize_score(score), float(reward))

    def select_population(
        self,
        population: List[Dict[str, Any]],
        target_population_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Keep the highest-scoring population members according to the configured selection rule.

        Old implementation:
        - states, resampled_idxs = resampler.resample(visited_states, self.num_agents, self.resampling)
        """
        ranked = sorted(population, key=lambda agent: agent["score"], reverse=True)

        if self.selection == "tournament":
            return self._tournament_select(ranked, target_population_size)

        return ranked[:target_population_size]

    async def adaptive_scale_population(
        self,
        idx: int,
        population: List[Dict[str, Any]],
        step: int,
        namespace: str,
        target_population_size: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Increase exploration when average quality is low and trim when it is high.

        Old implementation:
        - failed = [i for i, state in enumerate(states) if self.env.is_final(state)]
        - if visited_states != []:
        -     replacements, _ = resampler.resample(visited_states.copy(), len(failed), self.resampling)
        - else:
        -     replacements, _ = resampler.resample(
        -         [("", 1, state) for state in states],
        -         len(failed),
        -         resampling_method="linear"
        -     )
        - states = [replacements.pop(0) if i in failed else state for i, state in enumerate(states)]
        """
        if not population or not self.adaptive_scaling_enabled:
            return population, target_population_size

        avg_score = sum(agent["score"] for agent in population) / len(population)

        if avg_score < self.low_score_threshold and population:
            target_population_size = min(
                self.population_size,
                target_population_size + self.extra_agents,
            )
            best_state = max(population, key=lambda agent: agent["score"])["state"]
            additional_population = max(0, target_population_size - len(population))
            extra_seeds = [
                best_state.clone(randomness=random.randint(0, MAX_SEED))
                for _ in range(additional_population)
            ]
            extra_states = await self._advance_states(
                idx=idx,
                states=extra_seeds,
                step=step,
                namespace=namespace,
                phase="adaptive",
            )
            population.extend(
                {"state": extra_state, "score": 0.0}
                for extra_state in extra_states
            )
        elif avg_score > self.high_score_threshold:
            target_population_size = max(
                self.min_population,
                target_population_size // 2,
            )
            population = population[:target_population_size]

        return population, target_population_size

    async def _advance_states(
        self,
        idx: int,
        states: List[State],
        step: int,
        namespace: str,
        phase: str,
    ) -> List[State]:
        """
        Sample one action for each state and execute it in the environment.

        Old implementation:
        - action_coroutines = [
        -     self.step_agent.act(
        -         model=self.model,
        -         state=state,
        -         n=1,
        -         namespace=namespace,
        -         request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
        -         params=self.step_params
        -     )
        -     for i, state in enumerate(states)
        - ]
        - actions = await asyncio.gather(*action_coroutines)
        - states = [self.env.step(state, action[0]) for state, action in zip(states, actions)]
        """
        if not states:
            return []

        n_candidates = self.heterogeneous_candidates if self.heterogeneous_agents else 1
        action_coroutines = [
            self.step_agent.act(
                model=self.model,
                state=agent_state,
                n=n_candidates,
                namespace=namespace,
                request_id=f"idx{idx}-{phase}{step}-{hash(agent_state)}-agent{i}",
                params=self.step_params,
            )
            for i, agent_state in enumerate(states)
        ]
        actions = await asyncio.gather(*action_coroutines)

        next_states = []
        for agent_state, proposals in zip(states, actions):
            if proposals:
                proposal = random.choice(proposals) if self.heterogeneous_agents else proposals[0]
                next_states.append(self.env.step(agent_state, proposal))
            else:
                next_states.append(agent_state)
        return next_states

    def _tournament_select(
        self,
        ranked_population: List[Dict[str, Any]],
        target_population_size: int,
    ) -> List[Dict[str, Any]]:
        # Old implementation: no equivalent method existed.
        # Selection came from:
        # states, resampled_idxs = resampler.resample(visited_states, self.num_agents, self.resampling)
        pool = ranked_population.copy()
        selected = []

        while pool and len(selected) < target_population_size:
            sample_size = min(self.tournament_size, len(pool))
            contestants = random.sample(pool, sample_size)
            winner = max(contestants, key=lambda agent: agent["score"])
            selected.append(winner)
            pool.remove(winner)

        return selected

    def _normalize_score(self, score: Any) -> float:
        # Old implementation: no equivalent method existed.
        # Raw evaluator outputs were passed through directly, e.g.:
        # visited_states.append((f"{i}.{step}", value, state))
        if isinstance(score, (int, float)):
            return float(score)
        if isinstance(score, tuple) and score:
            return self._normalize_score(score[-1])
        if isinstance(score, list) and score:
            return self._normalize_score(score[-1])
        return 0.0
    
