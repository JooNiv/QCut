"""Tests for merging runs of gates on the same qubit pair before cutting."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, RZZGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutCZ, cutGate
from QCut.consolidate import consolidate_two_qubit_blocks, marker_gate
from QCut.qpd_generate import gamma
from QCut.qpd_operations import qpd_for_location

TOLERANCE = 0.1


def _unwrap(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace cut markers by the gates they carry, so the unitary can be compared."""
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        gate = marker_gate(instruction.operation)
        out.append(
            gate if gate is not None else instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )
    return out


def _equivalent(before: QuantumCircuit, after: QuantumCircuit) -> bool:
    return Operator(_unwrap(before)).equiv(Operator(_unwrap(after)))


def test_two_gates_on_a_pair_become_one():
    circuit = QuantumCircuit(2)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1}
    assert _equivalent(circuit, merged)


def test_marker_survives_the_merge():
    """A merged run on a marked pair must still be marked, or the cut is lost."""
    circuit = QuantumCircuit(2)
    circuit.append(**cutGate(RZZGate(0.4), 0, 1))
    circuit.rzz(0.4, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert list(merged.count_ops()) == ["CutUNITARY"]
    assert _equivalent(circuit, merged)


def test_lone_two_qubit_gate_keeps_its_name():
    """Absorbing single-qubit gates alone cannot help, and loses the named gate."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_single_qubit_gate_shared_by_two_pairs_is_not_applied_twice():
    """A gate on a shared qubit may only be absorbed into one of the two runs."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 0, 1)
    circuit.h(0)
    circuit.rzz(0.4, 0, 2)
    circuit.rzz(0.4, 0, 2)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 2}
    assert _equivalent(circuit, merged)


def test_a_wire_cut_ends_a_run():
    """Merging across a wire cut would move the cut, so the run has to stop there."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.4, 0, 1)
    circuit.append(cut(), [0])
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_a_gate_reaching_outside_the_pair_ends_a_run():
    circuit = QuantumCircuit(3)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 1, 2)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_a_disjoint_gate_does_not_end_a_run():
    """Gates on other qubits commute with the run, so they must not break it."""
    circuit = QuantumCircuit(4)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.9, 2, 3)
    circuit.rzz(0.4, 0, 1)
    merged = consolidate_two_qubit_blocks(circuit)
    assert merged.count_ops() == {"unitary": 1, "rzz": 1}
    assert _equivalent(circuit, merged)


def test_measurement_ends_a_run():
    circuit = QuantumCircuit(2, 1)
    circuit.rzz(0.4, 0, 1)
    circuit.measure(0, 0)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit) is circuit


def test_restrict_to_leaves_other_pairs_alone():
    circuit = QuantumCircuit(2)
    circuit.rzz(0.4, 0, 1)
    circuit.rzz(0.4, 0, 1)
    assert consolidate_two_qubit_blocks(circuit, restrict_to=set()) is circuit


def test_two_cz_cuts_collapse_to_a_free_cut():
    """CZ twice is the identity, so the merged cut costs gamma = 1."""
    circuit = QuantumCircuit(2)
    circuit.append(cutCZ(), [0, 1])
    circuit.append(cutCZ(), [0, 1])
    cut_circuit = ck.get_locations_and_subcircuits(circuit)
    assert len(cut_circuit.cut_locations) == 1
    qpd = qpd_for_location(cut_circuit.cut_locations[0])
    assert gamma(qpd) == pytest.approx(1.0)
    assert len(qpd) == 1


def _run(circuit, observables, options):
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    results = ck.run_experiments(experiment, backend=AerSimulator())
    values = ck.estimate_expectation_values(results, experiment.expv_data())
    return values, len(cut_circuit.cut_locations), experiment.num_groups


def test_merging_lowers_the_cost_and_keeps_the_answer():
    """Two marked rzz cost gamma 3.16 over 36 groups apart, 2.43 over 6 merged."""
    observables = SparsePauliOp(["IZ", "ZI", "ZZ"])

    def build():
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ry(0.7, 1)
        circuit.append(**cutGate(RZZGate(0.4), 0, 1))
        circuit.append(**cutGate(RZZGate(0.4), 0, 1))
        return circuit

    reference = QuantumCircuit(2)
    reference.h(0)
    reference.ry(0.7, 1)
    reference.rzz(0.8, 0, 1)
    state = Statevector(reference)
    exact = [float(np.real(state.expectation_value(p))) for p in observables.paulis]

    off, cuts_off, groups_off = _run(
        build(), observables, CutOptions(consolidate=False)
    )
    on, cuts_on, groups_on = _run(build(), observables, CutOptions(consolidate=True))

    assert (cuts_off, groups_off) == (2, 36)
    assert (cuts_on, groups_on) == (1, 6)
    for expected, without, with_ in zip(exact, off, on):
        assert abs(expected - without) < TOLERANCE
        assert abs(expected - with_) < TOLERANCE


def test_find_cuts_consolidates_before_partitioning():
    """A merged run is one graph edge to cut rather than several."""
    circuit = QuantumCircuit(4)
    for qubit in range(4):
        circuit.ry(0.4 + 0.2 * qubit, qubit)
    circuit.rzz(0.3, 0, 1)
    circuit.rzz(0.4, 1, 2)
    circuit.rzz(0.4, 1, 2)
    circuit.rzz(0.5, 2, 3)

    with_merge = ck.find_cuts(circuit.copy(), num_partitions=2)
    without = ck.find_cuts(
        circuit.copy(), num_partitions=2, options=CutOptions(consolidate=False)
    )
    observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "ZIII"])
    state = Statevector(circuit)
    exact = [float(np.real(state.expectation_value(p))) for p in observables.paulis]

    for cut_circuit in (with_merge, without):
        experiment = ck.get_experiment_circuits(cut_circuit, observables)
        results = ck.run_experiments(experiment, backend=AerSimulator())
        values = ck.estimate_expectation_values(results, experiment.expv_data())
        for expected, actual in zip(exact, values):
            assert abs(expected - actual) < TOLERANCE


def test_options_reach_the_cut_circuit():
    options = CutOptions(consolidate=False)
    circuit = QuantumCircuit(2)
    circuit.append(cutCZ(), [0, 1])
    cut_circuit = ck.get_locations_and_subcircuits(circuit, options=options)
    assert cut_circuit.options is options
    experiment = ck.get_experiment_circuits(cut_circuit, SparsePauliOp(["IZ", "ZI"]))
    assert experiment.options is options


def test_marker_gate_recovers_the_per_gate_markers():
    assert marker_gate(cutCZ()).name == CZGate().name
    assert marker_gate(cut()) is None
