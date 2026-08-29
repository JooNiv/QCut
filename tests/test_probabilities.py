"""Reconstructing a distribution from the Z expectation values of a cut circuit."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import cutGate
from QCut.execution.probabilities import QuasiProbabilities, _all_z_paulis_for_subset

SHOTS = 40000


def _bell_with_a_gate_cut():
    """A cut Bell pair with a spectator, so the marginal is not the whole state."""
    marked, plain = QuantumCircuit(3), QuantumCircuit(3)
    marked.h(0)
    plain.h(0)
    marked.append(**cutGate(CXGate(), 0, 1))
    plain.cx(0, 1)
    for circuit in (marked, plain):
        circuit.x(2)
    return marked, plain


def _spread_state():
    """Distinct marginals on every qubit, so a mis-ordered key cannot pass."""
    marked, plain = QuantumCircuit(3), QuantumCircuit(3)
    for circuit in (marked, plain):
        circuit.ry(0.7, 0)
    marked.append(**cutGate(CXGate(), 0, 1))
    plain.cx(0, 1)
    for circuit in (marked, plain):
        circuit.ry(1.3, 2)
    return marked, plain


def _reconstruct(marked, qubits, shots=SHOTS):
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), qubits=qubits
    )
    results = ck.run_experiments(experiment, shots=shots, backend=AerSimulator())
    return ck.estimate_probabilities(results)


def _exact_marginal(plain, qubits):
    """The true distribution over ``qubits``, keyed the way QCut keys it."""
    marginal: dict[str, float] = {}
    for bits, probability in Statevector(plain).probabilities_dict().items():
        per_qubit = bits[::-1]
        key = "".join(per_qubit[qubit] for qubit in reversed(qubits))
        marginal[key] = marginal.get(key, 0.0) + probability
    return marginal


@pytest.mark.sim
def test_the_reconstruction_matches_the_exact_distribution():
    marked, plain = _bell_with_a_gate_cut()
    probs = _reconstruct(marked, [0, 1])
    exact = _exact_marginal(plain, [0, 1])

    assert set(probs) == {"00", "01", "10", "11"}
    for key, value in probs.nearest_probabilities().items():
        assert value == pytest.approx(exact.get(key, 0.0), abs=0.05)


@pytest.mark.sim
def test_a_subset_reconstructs_the_marginal_over_those_qubits():
    """The chosen qubits need not be contiguous, and the rest are summed over."""
    marked, plain = _spread_state()
    probs = _reconstruct(marked, [0, 2])
    exact = _exact_marginal(plain, [0, 2])

    for key, value in probs.nearest_probabilities().items():
        assert value == pytest.approx(exact.get(key, 0.0), abs=0.05)


@pytest.mark.sim
def test_the_first_qubit_given_is_the_last_character_of_the_key():
    """Keys follow the order ``qubits`` was given, not the circuit's own order."""
    marked, plain = _spread_state()
    forwards = _reconstruct(marked, [0, 2]).nearest_probabilities()
    backwards = _reconstruct(marked, [2, 0]).nearest_probabilities()

    for key, value in forwards.items():
        assert backwards[key[::-1]] == pytest.approx(value, abs=0.05)

    exact = _exact_marginal(plain, [0, 2])
    assert forwards["10"] == pytest.approx(exact["10"], abs=0.05)


@pytest.mark.sim
def test_the_values_are_quasi_probabilities_and_the_projection_is_not():
    marked, _plain = _bell_with_a_gate_cut()
    probs = _reconstruct(marked, [0, 1], shots=2000)

    assert isinstance(probs, dict)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)

    nearest = probs.nearest_probabilities()
    assert min(nearest.values()) >= 0.0
    assert sum(nearest.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(nearest) == set(probs), "clipped bitstrings are kept, as zeros"
    assert probs.quasi_probabilities() == dict(probs)


def test_the_total_is_one_however_wrong_the_estimates_are():
    """The identity term fixes the sum, so the total cannot be used as a check."""
    wrong = np.array([1 + 0.5 - 0.5, 1 - 0.5 + 0.5, 1 + 0.5 + 0.5, 1 - 0.5 - 0.5]) / 4
    nonsense = QuasiProbabilities(dict(zip(["00", "01", "10", "11"], wrong)))

    assert sum(nonsense.values()) == pytest.approx(1.0)


@pytest.mark.sim
def test_counts_scale_by_the_shots_the_experiment_ran_at():
    marked, _plain = _bell_with_a_gate_cut()
    probs = _reconstruct(marked, [0, 1], shots=2000)

    assert probs.shots == 2000
    assert sum(probs.counts().values()) == pytest.approx(2000, abs=1e-6)
    assert sum(probs.counts(shots=137).values()) == pytest.approx(137, abs=1e-6)
    assert min(probs.counts().values()) >= 0.0, "counts are projected first"


def test_counts_need_a_shot_count_from_somewhere():
    bare = QuasiProbabilities({"0": 0.6, "1": 0.4})

    with pytest.raises(ValueError, match="no shot count"):
        bare.counts()
    assert bare.counts(shots=10) == {"0": 6.0, "1": 4.0}


@pytest.mark.sim
def test_every_z_string_shares_one_measurement_setting():
    """The observables multiply with the qubit count, the circuits do not."""
    marked, _plain = _spread_state()
    counts = []
    for width in (1, 2, 3):
        experiment = ck.get_experiment_circuits(
            ck.get_locations_and_subcircuits(marked), qubits=list(range(width))
        )
        assert len(experiment.observables) == 2**width - 1
        assert experiment.num_obs_groups == 1
        counts.append(experiment.num_circuits)

    assert len(set(counts)) == 1, f"circuit count should not grow: {counts}"


def test_probabilities_need_an_experiment_built_from_qubits():
    marked, _plain = _bell_with_a_gate_cut()
    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked), observables=SparsePauliOp(["IIZ"])
    )
    results = ck.run_experiments(experiment, shots=128, backend=AerSimulator())

    assert not experiment.can_reconstruct_probabilities
    with pytest.raises(ValueError, match="Cannot reconstruct probabilities"):
        ck.estimate_probabilities(results)


def test_observables_and_qubits_are_mutually_exclusive():
    marked, _plain = _bell_with_a_gate_cut()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    with pytest.raises(ValueError, match="Either observables or qubits"):
        ck.get_experiment_circuits(cut_circuit)
    with pytest.raises(ValueError, match="Only one of observables or qubits"):
        ck.get_experiment_circuits(
            cut_circuit, observables=SparsePauliOp(["IIZ"]), qubits=[0]
        )


@pytest.mark.parametrize(
    ("qubits", "message"),
    [
        ([], "at least one qubit"),
        ([0, 0], "must not repeat"),
        ([3], "outside the 3-qubit circuit"),
        ([-1], "outside the 3-qubit circuit"),
    ],
)
def test_the_qubits_asked_for_have_to_exist(qubits, message):
    """A qubit off the end used to wrap round onto a different one."""
    with pytest.raises(ValueError, match=message):
        _all_z_paulis_for_subset(3, qubits)


def test_a_circuit_with_measurements_is_accepted_and_left_alone():
    """Final measurements are stripped before splitting, on a copy."""
    marked, _plain = _bell_with_a_gate_cut()
    marked.measure_all()
    before = dict(marked.count_ops())

    ck.get_locations_and_subcircuits(marked)

    assert dict(marked.count_ops()) == before
