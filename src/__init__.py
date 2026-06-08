from typing import TypedDict
from .typedefs import DecodingParameters

class BenchmarkFactory:
    registry = {}

    @classmethod
    def register(cls, benchmark_cls):
        cls.registry[benchmark_cls.__name__.lower()] = benchmark_cls
        return benchmark_cls
    
    @classmethod
    def get(cls, task: str, *args, **kwargs):
        key = f"benchmark{task}".lower()
        try:
            return cls.registry[key](path=f"datasets/dataset_{task}.csv.gz",*args, **kwargs)
        except KeyError:
            fallback_key = key.replace("_", "")
            try:
                return cls.registry[fallback_key](path=f"datasets/dataset_{task}.csv.gz",*args, **kwargs)
            except KeyError:
                raise ValueError(f"No benchmark found for task={task}")
        
class EnvironmentFactory:
    registry = {}

    @classmethod
    def register(cls, env_cls):
        cls.registry[env_cls.__name__.lower()] = env_cls
        return env_cls
    
    @classmethod
    def get(cls, task: str, *args, **kwargs):
        key = f"environment{task}".lower()
        try:
            return cls.registry[key](*args, **kwargs)
        except KeyError:
            fallback_key = key.replace("_", "")
            try:
                return cls.registry[fallback_key](*args, **kwargs)
            except KeyError:
                raise ValueError(f"No environment found for task={task}")
    
class AgentFactory:
    registry = {}

    @classmethod
    def register(cls, agent_cls):
        cls.registry[agent_cls.__name__.lower()] = agent_cls
        return agent_cls

    @classmethod
    def get(cls, agent_type: str, benchmark: str, *args, **kwargs):
        key = f"agent{agent_type}{benchmark}".lower()
        try:
            return cls.registry[key]#(*args, **kwargs) : Not initialized
        except KeyError:
            fallback_key = key.replace("_", "")
            try:
                return cls.registry[fallback_key]
            except KeyError:
                raise ValueError(f"No agent found for type={agent_type}, benchmark={benchmark}")

class AgentDictFactory:
    registry = {}

    @classmethod
    def register(cls, agent_dict_cls):
        cls.registry[agent_dict_cls.__name__.lower()] = agent_dict_cls
        return agent_dict_cls

    @classmethod
    def get(cls, method: str, *args, **kwargs):
        key = f"agentdict{method}".lower()
        try:
            return cls.registry[key](*args, **kwargs)
        except KeyError:
            raise ValueError(f"No agent dict found for method={method}")


class MethodFactory:
    registry = {}

    @classmethod
    def register(cls, method_cls):
        key = method_cls.__name__.lower()
        cls.registry[key] = method_cls
        cls.registry[key.replace("_", "")] = method_cls
        return method_cls

    @classmethod
    def _build_step_agent_specs(
        cls,
        benchmark: str,
        params: DecodingParameters,
        config,
        default_agent_type: str,
        default_count: int,
    ):
        step_agent_configs = getattr(config, "step_agents", None)
        if not step_agent_configs:
            return [
                {
                    "agent_type": default_agent_type,
                    "agent": AgentFactory.get(default_agent_type, benchmark),
                    "params": params,
                    "num_agents": int(default_count),
                }
            ]

        specs = []
        for step_agent_config in step_agent_configs:
            agent_type = step_agent_config.get("agent_type", default_agent_type)
            specs.append(
                {
                    "agent_type": agent_type,
                    "agent": AgentFactory.get(agent_type, benchmark),
                    "params": params,
                    "num_agents": int(step_agent_config.get("num_agents", 1)),
                }
            )
        return specs

    @classmethod
    def get(cls, method: str, benchmark: str, params: DecodingParameters, *args, **kwargs):
        key = f"method{method}".lower()
        config = kwargs.get("config")
        method_key = method.lower()

        
        if method_key == "io":
            agents = {
                "step": AgentFactory.get("io", benchmark),
            }
        elif method_key in ["cot", "cot_sc"]:
            agents = {
                "step": AgentFactory.get("cot", benchmark),
            }
        elif method_key == "foa":
            agents = {
                "step": AgentFactory.get("act", benchmark),
                "evaluate": AgentFactory.get("evaluate", benchmark),
            }
        elif method_key in ["tot_bfs", "tot_dfs"]:
            agents = {
                "step": AgentFactory.get("bfs", benchmark),
                "evaluate": AgentFactory.get("evaluate", benchmark),
            }
        elif method_key == "got":
            agents = {
                "step": AgentFactory.get("act", benchmark),
                "aggregate": AgentFactory.get("aggregate", benchmark),
                "evaluate": AgentFactory.get("evaluate", benchmark),
            }
        elif method_key == "rap":
            agents = {
                "step": AgentFactory.get("react", benchmark),
                "evaluate": AgentFactory.get("selfevaluate", benchmark),
            }
        elif method_key == "react":
            agents = {
                "step": AgentFactory.get("react", benchmark),
            }
        elif method_key == "heterogeneous_foa":
            agents = {
                "evaluate": AgentFactory.get("evaluate", benchmark),
                "step_agents": cls._build_step_agent_specs(
                    benchmark=benchmark,
                    params=params,
                    config=config,
                    default_agent_type="act",
                    default_count=getattr(config, "num_agents", 1),
                ),
            }
        elif method_key in ["reagents", "reagents_v2comp", "reagents_v2tour", "reagents_v3"]:
            agents = {
                "evaluate": AgentFactory.get("evaluate", benchmark),
                "step_agents": cls._build_step_agent_specs(
                    benchmark=benchmark,
                    params=params,
                    config=config,
                    default_agent_type="act",
                    default_count=getattr(config, "width", 1),
                ),
                "difficulty_agent": (
                    {
                        "agent": AgentFactory.get(getattr(config, "difficulty_agent_type", "population"), benchmark),
                        "params": params,
                    }
                    if getattr(config, "difficulty_agent_type", None)
                    else None
                ),
            }
        elif method_key == "reagents_aos" or method_key == "reagents_alns":
            if getattr(config, "step_agents", None):
                step_agents = cls._build_step_agent_specs(
                    benchmark=benchmark,
                    params=params,
                    config=config,
                    default_agent_type="act",
                    default_count=getattr(config, "width", 1),
                )
            else:
                step_agents = [
                    {
                        "agent_type": "act",
                        "agent": AgentFactory.get("act", benchmark),
                        "params": params,
                        "num_agents": 1,
                    },
                    {
                        "agent_type": "react",
                        "agent": AgentFactory.get("react", benchmark),
                        "params": params,
                        "num_agents": 1,
                    },
                ]

            agents = {
                "evaluate": AgentFactory.get("evaluate", benchmark),
                "step_agents": step_agents,
                "difficulty_agent": (
                    {
                        "agent": AgentFactory.get(getattr(config, "difficulty_agent_type", "population"), benchmark),
                        "params": params,
                    }
                    if getattr(config, "difficulty_agent_type", None)
                    else None
                ),
            }
        else:
            raise NotImplementedError(f"Method {method} is not implemented yet.")
        
        # For the moment, only supporting same params for all agents
        agents.update({k+"_params": params for k in agents.keys() if k not in ["step_agents", "difficulty_agent"]})

        try:
            return cls.registry[key](agents=agents, *args, **kwargs)
        except KeyError:
            fallback_key = key.replace("_", "")
            try:
                return cls.registry[fallback_key](agents=agents, *args, **kwargs)
            except KeyError:
                raise ValueError(f"No method found for name={method}")
