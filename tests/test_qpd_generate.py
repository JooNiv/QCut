"""Tests for on-the-fly QPD generation for arbitrary two-qubit gates."""

import numpy as np
import pytest
from qiskit.circuit import Measure, QuantumCircuit
from qiskit.circuit.library import (
    CPhaseGate,
    CRXGate,
    CRZGate,
    CXGate,
    CZGate,
    DCXGate,
    ECRGate,
    RZZGate,
    SwapGate,
    UnitaryGate,
    XXPlusYYGate,
    iSwapGate,
)
from qiskit.quantum_info import Operator, random_unitary

from QCut.errors.qcuterror import QCutError
from QCut.qpd.qpd import cz_qpd, iswap_qpd, swap_qpd
from QCut.qpd.qpd_generate import (
    gamma,
    gamma_optimal,
    optimality_gap,
    qpd_from_gate,
    qpd_from_u,
    u_from_kak,
)

_SUP = lambda m: np.kron(m, np.conj(m))  # noqa: E731
_MEAS = _SUP(np.diag([1, 0]).astype(complex)) - _SUP(np.diag([0, 1]).astype(complex))


def _primitive_superop(circuit: QuantumCircuit) -> np.ndarray:
    """4x4 superoperator of a QPD primitive.

    A ``measure`` becomes the +/-1 weighted pair of computational-basis projectors,
    matching how QCut folds the measured bit into the estimator.
    """
    out = np.eye(4, dtype=complex)
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            out = _MEAS @ out
        else:
            out = _SUP(Operator(instruction.operation).data) @ out
    return out


def _channel_from_qpd(qpd: list[dict]) -> np.ndarray:
    """Reassemble the 16x16 two-qubit channel a QPD represents."""
    total = np.zeros((16, 16), dtype=complex)
    for term in qpd:
        s_0 = _primitive_superop(term["op_0"]).reshape(2, 2, 2, 2)
        s_1 = _primitive_superop(term["op_1"]).reshape(2, 2, 2, 2)
        # op_0 acts on qubit 0, the right kron factor.
        joint = np.einsum("acbd,ACBD->aAcCbBdD", s_1, s_0).reshape(16, 16)
        total += term["c"] * joint
    return total


def _channel_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.kron(matrix, np.conj(matrix))


NAMED_GATES = [
    ("cz", CZGate()),
    ("cx", CXGate()),
    ("swap", SwapGate()),
    ("iswap", iSwapGate()),
    ("dcx", DCXGate()),
    ("ecr", ECRGate()),
    ("rzz", RZZGate(0.3)),
    ("crz", CRZGate(0.8)),
    ("crx", CRXGate(1.9)),
    ("cphase", CPhaseGate(0.7)),
    ("xx_plus_yy", XXPlusYYGate(0.9, 0.4)),
]


@pytest.mark.parametrize(("name", "gate"), NAMED_GATES, ids=[n for n, _ in NAMED_GATES])
def test_generated_qpd_reconstructs_the_channel(name, gate):
    """The generated table must sum to exactly the gate's channel.

    Pins both the KAK sign convention and the plus/minus labelling of the primitives.
    """
    qpd = qpd_from_gate(gate)
    error = np.abs(
        _channel_from_qpd(qpd) - _channel_from_matrix(gate.to_matrix())
    ).max()
    assert error < 1e-12, f"{name}: reconstruction error {error:.2e}"


@pytest.mark.parametrize("seed", range(10))
def test_generated_qpd_reconstructs_random_su4(seed):
    """Generic two-qubit unitaries need all 58 rows, including the imaginary family."""
    matrix = random_unitary(4, seed=1234 + seed).data
    qpd = qpd_from_gate(UnitaryGate(matrix))
    error = np.abs(_channel_from_qpd(qpd) - _channel_from_matrix(matrix)).max()
    assert error < 1e-12, f"seed {seed}: reconstruction error {error:.2e}"
    assert len(qpd) <= 58


def _row_multiset(qpd: list[dict]) -> list[tuple]:
    """Order-independent fingerprint of a QPD, keyed on superoperators.

    Two primitives can implement the same Kraus map through different gate sequences.
    """
    rows = []
    for term in qpd:
        key = (
            np.round(_primitive_superop(term["op_0"]), 9).tobytes(),
            np.round(_primitive_superop(term["op_1"]), 9).tobytes(),
            round(float(term["c"]), 9),
        )
        rows.append(key)
    return sorted(rows)


def test_generated_matches_the_hand_written_swap_table():
    """Generation must reproduce swap_qpd exactly, row for row.

    SWAP is the only case where this is meaningful, since its local unitaries are
    trivial and the generated table lands in the frame the derivation used. cz and iswap
    come out in a different but equivalent frame.
    """
    generated = qpd_from_gate(SwapGate())
    assert len(generated) == len(swap_qpd)
    assert _row_multiset(generated) == _row_multiset(swap_qpd)


@pytest.mark.parametrize(
    ("gate", "hand_table"),
    [(CZGate(), cz_qpd), (SwapGate(), swap_qpd), (iSwapGate(), iswap_qpd)],
    ids=["cz", "swap", "iswap"],
)
def test_hand_written_tables_agree_with_generation(gate, hand_table):
    """The hand tables and the generator must have the same row count and gamma."""
    generated = qpd_from_gate(gate)
    assert len(generated) == len(hand_table)
    assert gamma(generated) == pytest.approx(gamma(hand_table))


@pytest.mark.parametrize(
    ("gate", "hand_table"),
    [(CZGate(), cz_qpd), (SwapGate(), swap_qpd), (iSwapGate(), iswap_qpd)],
    ids=["cz", "swap", "iswap"],
)
def test_hand_written_tables_reconstruct_their_channel(gate, hand_table):
    """The hand-derived tables in QCut.qpd must sum to the channel they claim.

    Guards the plus/minus labelling of the primitives, which iswap_qpd's ayz rows once
    got wrong without any end-to-end test noticing.
    """
    error = np.abs(
        _channel_from_qpd(hand_table) - _channel_from_matrix(gate.to_matrix())
    ).max()
    assert error < 1e-12, f"{gate.name}: reconstruction error {error:.2e}"


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        (CZGate(), 3.0),
        (CXGate(), 3.0),
        (SwapGate(), 7.0),
        (iSwapGate(), 7.0),
        (DCXGate(), 7.0),
    ],
    ids=["cz", "cx", "swap", "iswap", "dcx"],
)
def test_gamma_of_named_gates(gate, expected):
    assert gamma(qpd_from_gate(gate)) == pytest.approx(expected)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.0, np.pi / 2, 2.5, np.pi])
def test_rzz_gamma_matches_the_closed_form(theta):
    """RZZ has the known optimum gamma = 1 + 2|sin(theta)|."""
    assert gamma(qpd_from_gate(RZZGate(theta))) == pytest.approx(
        1 + 2 * abs(np.sin(theta))
    )


@pytest.mark.parametrize(("name", "gate"), NAMED_GATES, ids=[n for n, _ in NAMED_GATES])
def test_named_gates_attain_the_optimal_gamma(name, gate):
    """Mitarai-Fujii is optimal for every gate anyone realistically cuts."""
    decomposition_u = _u_of(gate)
    assert optimality_gap(decomposition_u) == pytest.approx(0.0, abs=1e-9)
    assert gamma(qpd_from_gate(gate)) == pytest.approx(gamma_optimal(decomposition_u))


def _u_of(gate):
    from qiskit.synthesis import TwoQubitWeylDecomposition

    decomposition = TwoQubitWeylDecomposition(gate.to_matrix())
    return u_from_kak(decomposition.a, decomposition.b, decomposition.c)


@pytest.mark.parametrize("seed", range(5))
def test_generic_gates_stay_within_the_proven_bound(seed):
    """For generic SU(4) the construction runs above the optimum, but boundedly.

    Per pair ``|Re z| + |Im z| <= sqrt(2)|z|``, so ``gamma - 1`` can exceed the optimum
    by at most a factor of sqrt(2).
    """
    matrix = random_unitary(4, seed=99 + seed).data
    decomposition_u = _u_of(UnitaryGate(matrix))
    achieved = gamma(qpd_from_gate(UnitaryGate(matrix)))
    optimal = gamma_optimal(decomposition_u)
    assert achieved >= optimal - 1e-9
    assert achieved - 1 <= np.sqrt(2) * (optimal - 1) + 1e-9


def test_qpd_rows_do_not_alias_the_shared_primitives():
    """Generated rows must be copies, since the primitives are module singletons."""
    from QCut.qpd import qpd_gates

    before = len(qpd_gates.zmeas.data)
    qpd = qpd_from_gate(CRZGate(0.8))
    for term in qpd:
        term["op_0"].x(0)
    assert len(qpd_gates.zmeas.data) == before


def test_non_two_qubit_gate_is_rejected():
    with pytest.raises(QCutError, match="two-qubit"):
        qpd_from_gate(CZGate().control(1))


def test_unbound_parameter_gives_a_clear_error():
    from qiskit.circuit import Parameter

    with pytest.raises(QCutError, match="unbound parameters"):
        qpd_from_gate(RZZGate(Parameter("t")))


def test_bad_u_shape_is_rejected():
    with pytest.raises(QCutError, match=r"shape \(4,\)"):
        qpd_from_u(np.ones(3))
