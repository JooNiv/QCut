import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import cut, cutGate

circuit = QuantumCircuit(4)

mult = 1.635
circuit.r(mult * 0.46262, mult * 0.1446, 0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.cx(2, 3)


cut_circuit = QuantumCircuit(4)

mult = 1.635
cut_circuit.r(mult * 0.46262, mult * 0.1446, 0)
cut_circuit.append(**cutGate(CXGate(), 0, 1))
cut_circuit.append(cut(), [1])
cut_circuit.cx(1, 2)
cut_circuit.cx(2, 3)

subcirc_lens = [3, 4, 6]

res_expvs = [0.727323, 0.727323, 0.727323, 1.000000]


def test_cut_gate_num_subcircuits():
    cut_circ = cut_circuit.copy()

    cut_qc = ck.get_locations_and_subcircuits(cut_circ)

    assert cut_qc.num_subcircuits == 3


def test_cut_gate_subcircuits():
    cut_qc = ck.get_locations_and_subcircuits(cut_circuit.copy())

    for ind, circ in enumerate(cut_qc.subcircuits):
        assert len(circ.data) == subcirc_lens[ind]


def test_cut_gate_expectation_values():
    cut_qc = ck.get_locations_and_subcircuits(cut_circuit.copy())

    backend = AerSimulator()

    observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])

    cut_experiment = ck.get_experiment_circuits(cut_qc, observables)

    results = ck.run_experiments(cut_experiment, backend=backend)

    expectation_values = ck.estimate_expectation_values(
        results, cut_experiment.expv_data()
    )

    for ind, expv in enumerate(expectation_values):
        assert abs(expv - res_expvs[ind]) < 0.1


def test_cutGate_list_args():
    result_int = cutGate(CXGate(), 0, 1)
    result_list = cutGate(CXGate(), [0], [1])
    assert result_int["qargs"] == result_list["qargs"]
    assert result_int["instruction"].name == result_list["instruction"].name


def test_cutGate_single_qubit_raises():
    with pytest.raises(ValueError, match="at least 2 qubits"):
        cutGate(XGate(), 0, 1)


def test_cutGate_wrong_num_qargs_raises():
    with pytest.raises(ValueError, match="Expected 2 qubit arguments"):
        cutGate(CXGate(), [0, 1], [2])


def test_cutGate_duplicate_qargs_raises():
    with pytest.raises(ValueError, match="unique"):
        cutGate(CXGate(), 0, 0)


def test_cutGate_negative_qarg_raises():
    with pytest.raises(ValueError, match="non-negative"):
        cutGate(CXGate(), -1, 0)
