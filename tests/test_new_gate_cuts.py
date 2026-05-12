"""Tests for new gate cut functionality: cutSWAP, cutISWAP, CutLocation.gate_name,
QPD_REGISTRY, _get_weights, and end-to-end SWAP/ISWAP cut pipelines."""

import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import QCut as ck
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.postprocess import _get_weights
from QCut.qpd import cz_qpd, iswap_qpd, swap_qpd
from QCut.qpd_operations import QPD_REGISTRY, get_qpd_combinations


def test_cutSWAP_returns_instruction():
    instr = ck.cutSWAP()
    assert isinstance(instr, Instruction)
    assert instr.num_qubits == 2


def test_cutSWAP_can_be_appended():
    qc = QuantumCircuit(2)
    qc.append(ck.cutSWAP(), [0, 1])
    assert any("CutSWAP" in op.operation.name for op in qc.data)


def test_cutISWAP_returns_instruction():
    instr = ck.cutISWAP()
    assert isinstance(instr, Instruction)
    assert instr.num_qubits == 2


def test_cutISWAP_can_be_appended():
    qc = QuantumCircuit(2)
    qc.append(ck.cutISWAP(), [0, 1])
    assert any("CutISWAP" in op.operation.name for op in qc.data)

def _make_cut_location(gate_name=None):
    from qiskit import QuantumRegister

    qr = QuantumRegister(2)
    qubits = [(qr, 0), (qr, 1)]
    if gate_name is None:
        return CutLocation((qubits, 0))
    return CutLocation((qubits, 0), gate_name=gate_name)


def test_cut_location_default_gate_name():
    loc = _make_cut_location()
    assert loc.gate_name == "cz"


def test_cut_location_custom_gate_name():
    loc = _make_cut_location(gate_name="swap")
    assert loc.gate_name == "swap"


def test_cut_location_equality_same_gate():
    loc1 = _make_cut_location(gate_name="swap")
    loc2 = _make_cut_location(gate_name="swap")
    assert loc1 == loc2


def test_cut_location_equality_different_gate():
    loc_cz = _make_cut_location(gate_name="cz")
    loc_swap = _make_cut_location(gate_name="swap")
    assert loc_cz != loc_swap


def test_cut_location_str_includes_gate_name():
    loc = _make_cut_location(gate_name="iswap")
    assert "iswap" in str(loc)

def test_qpd_registry_contains_expected_gates():
    assert "cz" in QPD_REGISTRY
    assert "swap" in QPD_REGISTRY
    assert "iswap" in QPD_REGISTRY


def test_qpd_registry_maps_to_correct_lists():
    assert QPD_REGISTRY["cz"] is cz_qpd
    assert QPD_REGISTRY["swap"] is swap_qpd
    assert QPD_REGISTRY["iswap"] is iswap_qpd


def test_qpd_registry_cz_length():
    assert len(QPD_REGISTRY["cz"]) == 6


def test_qpd_registry_swap_length():
    assert len(QPD_REGISTRY["swap"]) == len(swap_qpd)


def test_qpd_registry_iswap_length():
    assert len(QPD_REGISTRY["iswap"]) == len(iswap_qpd)

def test_get_qpd_combinations_unknown_gate_raises():
    loc = _make_cut_location(gate_name="unknown_gate")
    with pytest.raises(NotImplementedError, match="unknown_gate"):
        list(get_qpd_combinations([loc]))


def test_get_qpd_combinations_cz_count():
    loc = _make_cut_location(gate_name="cz")
    combos = list(get_qpd_combinations([loc]))
    assert len(combos) == len(cz_qpd)


def test_get_qpd_combinations_swap_count():
    loc = _make_cut_location(gate_name="swap")
    combos = list(get_qpd_combinations([loc]))
    assert len(combos) == len(swap_qpd)


def test_get_qpd_combinations_iswap_count():
    loc = _make_cut_location(gate_name="iswap")
    combos = list(get_qpd_combinations([loc]))
    assert len(combos) == len(iswap_qpd)


def test_get_qpd_combinations_wire_cut_count():
    from QCut.qpd import identity_qpd

    wire_loc = SingleQubitCutLocation(((QuantumRegister(1), 0), 0))
    combos = list(get_qpd_combinations([wire_loc]))
    assert len(combos) == len(identity_qpd)


def test_get_qpd_combinations_mixed_cuts():
    from QCut.qpd import identity_qpd

    wire_loc = SingleQubitCutLocation(((QuantumRegister(1), 0), 0))
    gate_loc = _make_cut_location(gate_name="cz")
    combos = list(get_qpd_combinations([wire_loc, gate_loc]))
    assert len(combos) == len(identity_qpd) * len(cz_qpd)

def test_get_weights_sum_equals_num_groups():
    coefficients = [1 / 2, 1 / 2, -1 / 2, 1 / 2]
    num_groups = 4
    weights = list(_get_weights(coefficients, num_groups))
    assert abs(sum(weights) - num_groups) < 1e-9


def test_get_weights_proportional_to_abs_coeff():
    coefficients = [1.0, 2.0, 1.0]
    num_groups = 3
    weights = list(_get_weights(coefficients, num_groups))
    assert abs(weights[1] - 2 * weights[0]) < 1e-9
    assert abs(weights[2] - weights[0]) < 1e-9


def test_get_weights_zero_coefficients_raises():
    with pytest.raises(ValueError, match="zero"):
        list(_get_weights([0.0, 0.0], 2))


swap_circuit = QuantumCircuit(2)
swap_circuit.h(0)
swap_circuit.append(ck.cutSWAP(), [0, 1])

_swap_observables = SparsePauliOp(["IZ", "ZI"])
_swap_expected = [1.0, 0.0]


def test_swap_cut_num_groups():
    cut_qc = ck.get_locations_and_subcircuits(swap_circuit.copy())
    cut_exp = ck.get_experiment_circuits(cut_qc, _swap_observables)
    assert cut_exp.num_groups == len(swap_qpd)


def test_swap_cut_expectation_values():
    cut_qc = ck.get_locations_and_subcircuits(swap_circuit.copy())
    cut_exp = ck.get_experiment_circuits(cut_qc, _swap_observables)
    results = ck.run_experiments(cut_exp, backend=AerSimulator())
    expvs = ck.estimate_expectation_values(results, cut_exp.expv_data())
    for computed, expected in zip(expvs, _swap_expected):
        assert abs(computed - expected) < 0.15


iswap_circuit = QuantumCircuit(2)
iswap_circuit.h(0)
iswap_circuit.append(ck.cutISWAP(), [0, 1])

_iswap_observables = SparsePauliOp(["IZ", "ZI"])
_iswap_expected = [1.0, 0.0]


def test_iswap_cut_num_groups():
    cut_qc = ck.get_locations_and_subcircuits(iswap_circuit.copy())
    cut_exp = ck.get_experiment_circuits(cut_qc, _iswap_observables)
    assert cut_exp.num_groups == len(iswap_qpd)


def test_iswap_cut_expectation_values():
    cut_qc = ck.get_locations_and_subcircuits(iswap_circuit.copy())
    cut_exp = ck.get_experiment_circuits(cut_qc, _iswap_observables)
    results = ck.run_experiments(cut_exp, backend=AerSimulator())
    expvs = ck.estimate_expectation_values(results, cut_exp.expv_data())
    for computed, expected in zip(expvs, _iswap_expected):
        assert abs(computed - expected) < 0.15

def test_find_cuts_basis_includes_swap_and_iswap():
    from QCut.QCutFind.cut_finding import BASIS_GATES

    assert "swap" in BASIS_GATES
    assert "iswap" in BASIS_GATES


def test_find_cuts_swap_circuit():
    """find_cuts handles a circuit with SWAP gates without error."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.swap(1, 2)
    qc.swap(1, 2)
    qc.cx(2, 3)
    cut_qc = ck.find_cuts(qc, num_partitions=2, max_qubits=[2, 2], cuts="gate")
    assert len(cut_qc.subcircuits) == 2


def test_find_cuts_iswap_circuit():
    """find_cuts handles a circuit with ISWAP gates without error."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.iswap(1, 2)
    qc.iswap(1, 2)
    qc.iswap(1, 2)
    qc.cx(2, 3)
    cut_qc = ck.find_cuts(qc, num_partitions=2, max_qubits=[2, 2], cuts="gate")
    assert len(cut_qc.subcircuits) == 2
