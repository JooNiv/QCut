"""Tests for the unfolded analytic KAK decomposition of parametrised two-qubit gates."""

import numpy as np
import pytest
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CZGate,
    RXXGate,
    RYYGate,
    RZXGate,
    RZZGate,
    XXMinusYYGate,
    XXPlusYYGate,
)

from QCut.qcuterror import QCutError
from QCut.qpd_analytic import analytic_kak, reassemble
from QCut.qpd_generate import gamma, qpd_from_gate, qpd_from_u, u_from_kak

THETA = Parameter("t")

# Every family the automatic derivation is expected to handle, built with THETA free.
FAMILIES = [
    ("rzz", RZZGate(THETA)),
    ("rxx", RXXGate(THETA)),
    ("ryy", RYYGate(THETA)),
    ("rzx", RZXGate(THETA)),
    ("crz", CRZGate(THETA)),
    ("crx", CRXGate(THETA)),
    ("cry", CRYGate(THETA)),
    ("cphase", CPhaseGate(THETA)),
    ("xx_plus_yy", XXPlusYYGate(THETA, 0.4)),
    ("xx_minus_yy", XXMinusYYGate(THETA, -1.1)),
]
IDS = [name for name, _ in FAMILIES]

# Deliberately straddles the Weyl-chamber fold.
ANGLES = [0.2, 0.9, 1.7, 3.0, 4.4, 6.1]


def _bound(gate, theta):
    params = [theta if getattr(p, "parameters", None) else p for p in gate.params]
    probe = gate.copy()
    probe.params = params
    return probe


@pytest.mark.parametrize(("name", "gate"), FAMILIES, ids=IDS)
@pytest.mark.parametrize("theta", ANGLES)
def test_reassembly_matches_the_gate(name, gate, theta):
    """The unfolded decomposition must reproduce the gate at every angle."""
    decomposition = analytic_kak(gate)
    rebuilt = reassemble(decomposition, theta)
    target = _bound(gate, theta).to_matrix()
    error = np.abs(
        np.kron(rebuilt, rebuilt.conj()) - np.kron(target, target.conj())
    ).max()
    assert error < 1e-12, f"{name} at theta={theta}: channel error {error:.2e}"


@pytest.mark.parametrize(("name", "gate"), FAMILIES, ids=IDS)
def test_derivation_does_not_depend_on_the_probe_angle(name, gate, monkeypatch):
    """The derivation must not depend on where the generator was sampled.

    If probing at a different angle gave a different decomposition, it could not be
    baked into circuits before binding.
    """
    import QCut.qpd_analytic as module

    reference = analytic_kak(gate)
    for probe in (0.1, 0.35, 0.8, 1.3):
        monkeypatch.setattr(module, "REFERENCE_ANGLE", probe)
        again = analytic_kak(gate)
        # v_1/v_2 individually are not unique when two coordinates coincide, so only
        # the canonical class and the local generator have to agree exactly. The loop
        # below then checks that whichever basis was picked reproduces the gate.
        assert sorted(np.abs(again.coords)) == pytest.approx(
            sorted(np.abs(reference.coords)), abs=1e-9
        ), f"coordinates moved at probe {probe}"
        assert np.abs(again.h_local - reference.h_local).max() < 1e-9
        for theta in ANGLES:
            rebuilt = reassemble(again, theta)
            target = _bound(gate, theta).to_matrix()
            error = np.abs(
                np.kron(rebuilt, rebuilt.conj()) - np.kron(target, target.conj())
            ).max()
            assert error < 1e-12, f"probe {probe}, theta {theta}: error {error:.2e}"


@pytest.mark.parametrize(("name", "gate"), FAMILIES, ids=IDS)
def test_weyl_locals_do_vary_where_the_analytic_ones_do_not(name, gate):
    """Document why this module exists rather than reusing TwoQubitWeylDecomposition.

    The Weyl coordinate folds over the sampled range while the analytic one stays
    exactly linear.
    """
    from qiskit.synthesis import TwoQubitWeylDecomposition

    analytic = analytic_kak(gate)
    weyl_coords = []
    for theta in ANGLES:
        decomposition = TwoQubitWeylDecomposition(_bound(gate, theta).to_matrix())
        weyl_coords.append((decomposition.a, decomposition.b, decomposition.c))
        assert np.allclose(analytic.kak_coords(theta), -theta * analytic.coords)
    first = np.array([abs(c[0]) for c in weyl_coords])
    assert np.any(np.diff(first) < -1e-6), (
        f"{name}: expected the Weyl coordinate to fold over {ANGLES}, got {first}"
    )


@pytest.mark.parametrize(("name", "gate"), FAMILIES, ids=IDS)
def test_coordinates_are_linear_in_the_parameter(name, gate):
    """(a, b, c) = -theta * coords, with no piecewise folding."""
    decomposition = analytic_kak(gate)
    for theta in ANGLES:
        assert np.allclose(
            decomposition.kak_coords(theta), -theta * decomposition.coords
        )


@pytest.mark.parametrize(("name", "gate"), FAMILIES, ids=IDS)
@pytest.mark.parametrize("theta", ANGLES)
def test_generated_qpd_agrees_with_the_numeric_path(name, gate, theta):
    """A QPD built through the analytic route must match the bound-gate route.

    The two land in different frames, so individual rows differ, but the sampling cost
    and the row count must not.
    """
    decomposition = analytic_kak(gate)
    analytic_u = u_from_kak(*decomposition.kak_coords(theta))
    analytic_qpd = qpd_from_u(analytic_u)
    numeric_qpd = qpd_from_gate(_bound(gate, theta))
    assert gamma(analytic_qpd) == pytest.approx(gamma(numeric_qpd))
    assert len(analytic_qpd) == len(numeric_qpd)


def test_rzz_has_trivial_locals_and_the_expected_coordinate():
    """RZZ(theta) = exp(-i theta/2 ZZ): no locals, one coordinate of 1/2 per theta."""
    decomposition = analytic_kak(RZZGate(THETA))
    assert np.abs(decomposition.h_local).max() < 1e-12
    assert sorted(np.abs(decomposition.coords)) == pytest.approx([0.0, 0.0, 0.5])


def test_controlled_rotation_has_a_local_part():
    """CRZ carries a local RZ whose angle is linear in theta."""
    decomposition = analytic_kak(CRZGate(THETA))
    assert np.abs(decomposition.h_local).max() > 1e-9
    assert sorted(np.abs(decomposition.coords)) == pytest.approx([0.0, 0.0, 0.25])


def test_rzz_gamma_from_the_analytic_route_matches_the_closed_form():
    decomposition = analytic_kak(RZZGate(THETA))
    for theta in ANGLES:
        u = u_from_kak(*decomposition.kak_coords(theta))
        assert gamma(qpd_from_u(u)) == pytest.approx(1 + 2 * abs(np.sin(theta)))


def test_support_is_parameter_independent_for_generic_angles():
    """The row count must be fixed per family for num_groups to be known pre-bind."""
    for _, gate in FAMILIES:
        decomposition = analytic_kak(gate)
        counts = {
            len(qpd_from_u(u_from_kak(*decomposition.kak_coords(theta))))
            for theta in ANGLES
        }
        assert len(counts) == 1, f"{gate.name}: row count varies with theta ({counts})"


def test_bound_gate_is_rejected():
    with pytest.raises(QCutError, match="exactly one unbound parameter"):
        analytic_kak(RZZGate(0.3))


def test_two_unbound_parameters_are_rejected():
    with pytest.raises(QCutError, match="exactly one unbound parameter"):
        analytic_kak(XXPlusYYGate(Parameter("a"), Parameter("b")))


def test_non_two_qubit_gate_is_rejected():
    with pytest.raises(QCutError, match="two-qubit"):
        analytic_kak(CZGate().control(1))
