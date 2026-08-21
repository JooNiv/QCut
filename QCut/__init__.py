"""Init circuit knitting."""  # noqa: N999

from importlib.metadata import PackageNotFoundError, version

from QCut.backend_utility import (
    transpile_experiments,
    transpile_subcircuits,
)
from QCut.circuit_knitting import (
    get_experiment_circuits,
    run,
    run_cut_circuit,
    run_experiments,
)
from QCut.circuit_preparation import (
    get_locations_and_subcircuits,
)
from QCut.consolidate import consolidate_two_qubit_blocks
from QCut.options import DEFAULT_OPTIONS, CutOptions
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
    "cut",
    "cutCZ",
    "cutSWAP",
    "cutISWAP",
    "cutGate",
    "find_cuts",
    "CutOptions",
    "DEFAULT_OPTIONS",
    "consolidate_two_qubit_blocks",
]
