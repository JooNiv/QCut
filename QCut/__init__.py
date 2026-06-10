"""Init circuit knitting."""  # noqa: N999

from importlib.metadata import PackageNotFoundError, version

from QCut.backend_utility import (
    transpile_experiments,
    transpile_subcircuits,
)
from QCut.circuit_knitting import (
    get_experiment_circuits,
    mpi_run,
    run,
    run_cut_circuit,
    run_experiments,
)
from QCut.circuit_preparation import (
    get_locations_and_subcircuits,
)
from QCut.executors import (
    MPIExecutor,
    MultiprocessingExecutor,
    SerialExecutor,
    get_default_executor,
)
from QCut.postprocess import (
    estimate_expectation_values,
)
from QCut.QCutFind import find_cuts
from QCut.qpd_gates import cut, cutCZ, cutGate, cutISWAP, cutSWAP

try:
    __version__ = version("QCut")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "run_on_backend",
    "transpile_experiments",
    "transpile_subcircuits",
    "estimate_expectation_values",
    "get_experiment_circuits",
    "get_locations_and_subcircuits",
    "run",
    "run_cut_circuit",
    "run_experiments",
    "mpi_run",
    "SerialExecutor",
    "MultiprocessingExecutor",
    "MPIExecutor",
    "get_default_executor",
    "cut",
    "cutCZ",
    "cutSWAP",
    "cutISWAP",
    "cutGate",
    "find_cuts",
]
