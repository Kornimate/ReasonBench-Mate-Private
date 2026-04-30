from .agents import *
from .environment import EnvironmentMIMICRRS, EnvironmentMimic_RRS

try:
    from .benchmark import BenchmarkMIMICRRS, BenchmarkMimic_RRS
except ImportError as exc:
    from ... import BenchmarkFactory

    _benchmark_import_error = exc

    @BenchmarkFactory.register
    class BenchmarkMimic_RRS:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The MIMIC-RRS benchmark requires the optional HELM dependency "
                "(`helm.benchmark.scenarios.mimic_rrs_scenario`) and access to "
                "the configured MIMIC-RRS data path."
            ) from _benchmark_import_error

    BenchmarkMIMICRRS = BenchmarkMimic_RRS
