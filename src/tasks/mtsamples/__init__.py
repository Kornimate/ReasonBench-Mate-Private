from .agents import *
from .environment import EnvironmentMTSamples

try:
    from .benchmark import BenchmarkMTSamples
except ImportError as exc:
    from ... import BenchmarkFactory

    # Keep the environment and agents available when optional HELM benchmark
    # dependencies are not installed.
    _benchmark_import_error = exc

    @BenchmarkFactory.register
    class BenchmarkMTSamples:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The MTSamples benchmark requires the optional HELM dependency "
                "(`helm.benchmark.scenarios.mtsamples_procedures_scenario`)."
            ) from _benchmark_import_error
