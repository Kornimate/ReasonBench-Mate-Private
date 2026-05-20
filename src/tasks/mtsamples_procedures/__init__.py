from .agents import *
from .environment import EnvironmentMTSamplesProcedures

try:
    from .benchmark import BenchmarkMTSamplesProcedures
except ImportError as exc:
    from ... import BenchmarkFactory

    # Keep the environment and agents available when optional HELM benchmark
    # dependencies are not installed.
    _benchmark_import_error = exc

    @BenchmarkFactory.register
    class BenchmarkMTSamplesProcedures:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The MTSamplesProcedures benchmark requires the optional HELM dependency "
                "(`helm.benchmark.scenarios.mtsamples_procedures_scenario`)."
            ) from _benchmark_import_error
