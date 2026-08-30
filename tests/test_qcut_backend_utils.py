import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector

import QCut as ck
from QCut import cut, cutGate
from QCut.errors.qcuterror import QCutError
from QCut.execution.circuit_knitting import _backend_gate_names

cut_circ = QuantumCircuit(4)

mult = 1.635
cut_circ.r(mult * 0.46262, mult * 0.1446, 0)
cut_circ.append(**cutGate(CXGate(), 0, 1))
cut_circ.append(cut(), [1])
cut_circ.cx(1, 2)
cut_circ.cx(2, 3)

backend = GenericBackendV2(num_qubits=5, basis_gates=["cz", "r"])


def test_transpile_subcircuits():
    """Test transpilation of subcircuits with custom gates.

    This test verifies that the transpile_subcircuits function correctly transpiles
    subcircuits containing custom gates (cutCZ and cutGate) to a specified backend.
    It checks that the transpiled circuits contain the expected gates and that the
    custom gates are preserved in the transpilation process.
    """

    cut_copy = cut_circ.copy()

    cut_circuit = ck.get_locations_and_subcircuits(cut_copy)

    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)

    assert transpiled.num_subcircuits == 3

    assert transpiled.backend == backend

    basis = ["cz", "r"]

    for subcircuit in transpiled.subcircuits:
        for op in subcircuit.data:
            assert (
                op.operation.name.lower() in basis
                or "cut" in op.operation.name.lower()
                or "meas" in op.operation.name.lower()
                or "init" in op.operation.name.lower()
                or "obs" in op.operation.name.lower()
            )


def test_transpile_experiments():
    """Test transpilation of experiments with custom gates.

    This test verifies that the transpile_experiments function correctly transpiles
    experiments containing custom gates (cutCZ and cutGate) to a specified backend.
    It checks that the transpiled circuits contain the expected gates and that the
    custom gates are preserved in the transpilation process.
    """

    cut_copy = cut_circ.copy()

    cut_circuit = ck.get_locations_and_subcircuits(cut_copy)

    exp_circuits = ck.get_experiment_circuits(cut_circuit, SparsePauliOp("ZZZZ"))

    transpiled = ck.transpile_experiments(exp_circuits, backend, optimization_level=3)

    assert transpiled.backend == backend

    assert transpiled.num_circuits == exp_circuits.num_circuits

    basis = ["cz", "r"]

    for exp in transpiled.experiments:
        for exp_dict in exp:
            for circ in exp_dict.values():
                for op in circ.data:
                    assert (
                        op.operation.name.lower() in basis
                        or "cut" in op.operation.name.lower()
                        or "meas" in op.operation.name.lower()
                        or "init" in op.operation.name.lower()
                        or "obs" in op.operation.name.lower()
                    )


@pytest.mark.parametrize(
    "basis_gates",
    [None, ["cz", "r"], ["cx", "rz", "sx", "x"], ["ecr", "rz", "sx", "x"]],
)
def test_transpiled_subcircuits_build_experiments(basis_gates):
    """Build experiment circuits from subcircuits already transpiled to a backend.

    transpile_subcircuits attaches the backend to the CutCircuit it returns, which makes
    get_experiment_circuits write the observable basis changes in that backend's own
    gates. Asking a BackendV2 which gates those are used to go through an IQM only
    attribute, so this whole path raised AttributeError for any other BackendV2.
    """
    generic = GenericBackendV2(
        num_qubits=5,
        seed=11,
        **({} if basis_gates is None else {"basis_gates": basis_gates}),
    )

    cut_circuit = ck.get_locations_and_subcircuits(cut_circ.copy())
    transpiled = ck.transpile_subcircuits(cut_circuit, generic, optimization_level=3)

    experiments = ck.get_experiment_circuits(transpiled, SparsePauliOp("ZZZZ"))

    assert experiments.num_circuits > 0


def test_transpiled_subcircuits_give_correct_expectation_value():
    """The transpiled path reconstructs the same value the uncut circuit has."""
    generic = GenericBackendV2(num_qubits=5, seed=11)

    uncut = QuantumCircuit(4)
    uncut.h(0)
    uncut.cx(0, 1)
    uncut.cx(1, 2)
    uncut.cx(2, 3)
    exact = np.real(Statevector(uncut).expectation_value(SparsePauliOp("ZZZZ")))

    to_cut = QuantumCircuit(4)
    to_cut.h(0)
    to_cut.append(**cutGate(CXGate(), 0, 1))
    to_cut.append(cut(), [1])
    to_cut.cx(1, 2)
    to_cut.cx(2, 3)

    cut_circuit = ck.get_locations_and_subcircuits(to_cut)
    transpiled = ck.transpile_subcircuits(cut_circuit, generic, optimization_level=3)
    experiments = ck.get_experiment_circuits(transpiled, SparsePauliOp("ZZZZ"))
    results = ck.run_experiments(experiments, shots=20000)

    assert np.allclose(ck.estimate_expectation_values(results), exact, atol=0.1)


def test_backend_gate_names_reads_a_v2_target():
    """A BackendV2 with neither configuration() nor architecture answers by target."""
    generic = GenericBackendV2(num_qubits=5, basis_gates=["cz", "r"], seed=11)

    assert not hasattr(generic, "architecture")
    assert set(_backend_gate_names(generic)) >= {"cz", "r"}


def test_backend_gate_names_prefers_an_iqm_architecture():
    """An architecture is read before the target, as the prx rename expects."""

    class FakeArchitecture:
        gates = {"prx": None, "cz": None, "measure": None}

    class FakeIQMBackend:
        architecture = FakeArchitecture()
        target = GenericBackendV2(num_qubits=5, seed=11).target

    assert _backend_gate_names(FakeIQMBackend()) == ["prx", "cz", "measure"]


def test_backend_gate_names_rejects_a_backend_that_names_nothing():
    """A backend answering none of the three ways is reported, not left to crash."""

    class Nameless:
        pass

    with pytest.raises(QCutError, match="raise an issue"):
        _backend_gate_names(Nameless())
