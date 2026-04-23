from .agents import *
from .environment import EnvironmentPubMedQA
from ... import AgentFactory, BenchmarkFactory, EnvironmentFactory

try:
    from .benchmark import BenchmarkPubMedQA
except ImportError as exc:
    _benchmark_import_error = exc

    @BenchmarkFactory.register
    class BenchmarkPubMedQA:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The PubMedQA benchmark requires the optional HELM/MedHELM dependency "
                "(`helm.benchmark.scenarios.pubmed_qa_scenario`)."
            ) from _benchmark_import_error


EnvironmentFactory.registry["environmentpubmed_qa"] = EnvironmentPubMedQA

for agent_type in [
    "io",
    "cot",
    "act",
    "population",
    "bfs",
    "aggregate",
    "react",
    "evaluate",
    "selfevaluate",
]:
    underscore_key = f"agent{agent_type}pubmed_qa"
    compact_key = f"agent{agent_type}pubmedqa"
    if compact_key in AgentFactory.registry:
        AgentFactory.registry[underscore_key] = AgentFactory.registry[compact_key]

if "benchmarkpubmedqa" in BenchmarkFactory.registry:
    BenchmarkFactory.registry["benchmarkpubmed_qa"] = BenchmarkFactory.registry["benchmarkpubmedqa"]
