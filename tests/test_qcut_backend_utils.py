from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp

import QCut as ck
from QCut import cut, cutGate

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
