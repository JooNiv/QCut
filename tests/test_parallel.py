"""Running one experiment on several backends at once."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import ParallelBackend, cut, cutGate
from QCut.errors.qcuterror import QCutError


def _gate_cut():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.4 + 0.2 * qubit, qubit)
        circuit.cx(0, 1)
    marked.append(**cutGate(CXGate(), 1, 2))
    plain.cx(1, 2)
    for circuit in (marked, plain):
        circuit.cx(2, 3)
    return marked, plain, SparsePauliOp(["IIIZ", "IIZI", "ZIII"])


def _wide_gate_cut():
    """Halves of three qubits, so a two-qubit backend has no room for them."""
    marked, plain = QuantumCircuit(6), QuantumCircuit(6)
    for circuit in (marked, plain):
        for qubit in range(6):
            circuit.ry(0.4 + 0.2 * qubit, qubit)
        circuit.cx(0, 1)
        circuit.cx(3, 4)
    marked.append(**cutGate(CXGate(), 2, 3))
    plain.cx(2, 3)
    for circuit in (marked, plain):
        circuit.cx(4, 5)
    return marked, plain, SparsePauliOp(["IIIIIZ", "IIIIZI", "ZIIIII"])


def _communicating():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
    marked.append(cut(), [1])
    marked.append(cut(), [2])
    for circuit in (marked, plain):
        circuit.cx(1, 3)
        circuit.cx(2, 3)
    return marked, plain, SparsePauliOp(["IIIZ", "IIZI", "IZII"])


def _experiment(marked, observables):
    return ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), observables
    )


def _exact(plain, observables):
    return np.array(
        [
            float(np.real(Statevector(plain).expectation_value(pauli)))
            for pauli in observables.paulis
        ]
    )


@pytest.mark.sim
@pytest.mark.parametrize("case", [_gate_cut, _communicating])
def test_the_work_is_shared_and_the_answers_are_the_same(case):
    """Both execution paths, since the communicating one submits in waves."""
    marked, plain, observables = case()
    experiment = _experiment(marked, observables)
    backend = ParallelBackend([AerSimulator(), AerSimulator()])

    values = np.array(
        ck.estimate_expectation_values(
            ck.run_experiments(
                experiment, shots=20000, backend=backend, max_batch_size=8
            )
        )
    )

    assert all(count > 0 for count in backend.submitted), "one was never used"
    assert np.allclose(values, _exact(plain, observables), atol=0.05)


@pytest.mark.sim
def test_a_backend_with_no_room_is_left_out():
    """A fleet may hold machines too small for some of the pieces."""
    marked, plain, observables = _wide_gate_cut()
    experiment = _experiment(marked, observables)
    narrow, wide = GenericBackendV2(2, seed=3), GenericBackendV2(9, seed=4)
    backend = ParallelBackend([narrow, wide])

    values = np.array(
        ck.estimate_expectation_values(
            ck.run_experiments(
                experiment, shots=20000, backend=backend, max_batch_size=8
            )
        )
    )

    assert backend.submitted[0] == 0, "the narrow backend was given circuits"
    assert backend.submitted[1] == experiment.num_circuits
    assert np.allclose(values, _exact(plain, observables), atol=0.1)


def test_nothing_wide_enough_says_so():
    marked, _plain, observables = _wide_gate_cut()
    experiment = _experiment(marked, observables)
    backend = ParallelBackend(
        [GenericBackendV2(2, seed=1), GenericBackendV2(2, seed=2)]
    )

    with pytest.raises(QCutError, match="has room for"):
        ck.run_experiments(experiment, shots=128, backend=backend)


def test_one_backend_is_not_a_fleet():
    with pytest.raises(QCutError, match="at least two backends"):
        ParallelBackend([AerSimulator()])


def test_the_shot_cap_is_the_one_every_backend_can_meet():
    class Capped(AerSimulator):
        def __init__(self, cap):
            super().__init__()
            self.max_shots = cap

    assert ParallelBackend([Capped(4096), Capped(1024)]).max_shots == 1024
    # None when nothing declares one, which is what leaves the shots uncapped
    assert ParallelBackend([AerSimulator(), AerSimulator()]).max_shots is None
