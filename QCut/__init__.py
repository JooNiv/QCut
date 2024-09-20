"""Init circuit knitting."""  # noqa: N999

from importlib.metadata import PackageNotFoundError, version

from QCut.backend_utility import (
    expectation_values,
    run_and_expectation_value,
    run_on_backend,
    transpile_experiments,
)
from QCut.helper import get_pauli_list
from QCut.identity_qpd import identity_qpd
from QCut.qpd_gates import cut_wire
from QCut.wirecut import (
    estimate_expectation_values,
    get_experiment_circuits,
    get_locations_and_subcircuits,
    run,
    run_cut_circuit,
    run_experiments,
)

try:
    __version__ = version("QCut")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "expectation_values",
    "run_on_backend",
    "run_and_expectation_value",
    "transpile_experiments",
    "estimate_expectation_values",
    "get_experiment_circuits",
    "get_locations_and_subcircuits",
    "get_pauli_list",
    "run",
    "run_cut_circuit",
    "run_experiments",
    "cut_wire",
    "identity_qpd",
]
