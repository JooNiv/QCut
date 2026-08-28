"""Unfolded KAK decomposition for parametrised two-qubit gates.

:func:`QCut.qpd_generate.qpd_from_gate` needs a numeric matrix, and it could not be made
to work before binding. It goes through ``TwoQubitWeylDecomposition``, which folds the
interaction coordinates into the chamber :math:`\\pi/4 \\ge a \\ge b \\ge |c|`. That
fold is piecewise in the gate parameter and the locals absorb it discontinuously, so
circuits built from Weyl output cannot be fixed ahead of binding.

The fold only exists to make the canonical form unique, which the Mitarai-Fujii
construction does not need. This module recovers an unfolded decomposition whose local
unitaries are constant and whose coordinates are linear in the parameter.

1. Extract the generator at one reference angle, :math:`H = i \\log U(\\theta_0)`. Valid
   for all :math:`\\theta` because the supported families are one-parameter groups.
2. Expand :math:`H` in the two-qubit Pauli basis and split off the terms carrying an
   identity factor. They commute with the rest, so the exponential factorises.
3. Canonicalise the non-local part, a real 3x3 matrix, by an SVD whose factors are
   forced into :math:`SO(3)` and lifted to :math:`SU(2)`.

The result is parameter-independent locals plus coordinates :math:`(a, b, c) =
-\\theta d`, so one decomposition per gate type serves every parameter value. Where two
coordinates coincide the locals are not unique, which is harmless because
:func:`analytic_kak` verifies whichever basis it picked before returning it.

No per-family hand derivation is involved. The steps are generic and are verified
against ten rotation and controlled-rotation families in ``tests/test_qpd_analytic.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import Gate

from QCut.errors.qcuterror import QCutError

logger: logging.Logger = logging.getLogger(__name__)

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = (_I, _X, _Y, _Z)

#: Angle at which the generator is sampled. Small enough that every supported family
#: keeps its eigenphases on the principal branch of the matrix logarithm, but large
#: enough not to amplify round-off.
REFERENCE_ANGLE: float = 0.5

_ATOL: float = 1e-9


@dataclass(frozen=True)
class AnalyticKAK:
    """An unfolded KAK decomposition valid for every value of the gate parameter.

    The gate at parameter ``theta`` is, up to global phase::

        exp(-i * theta * h_local)
          @ (v_1 (x) v_2)^dag @ exp(i * (a XX + b YY + c ZZ)) @ (v_1 (x) v_2)

    with ``(a, b, c) = -theta * coords``. ``v_1`` acts on the kron-left qubit and
    ``v_2`` on the kron-right one, ``coords`` are the canonical coordinates per unit
    parameter, ``h_local`` is the local part of the generator, and ``free_index`` says
    which of the gate's parameters the decomposition is linear in.
    """

    v_1: np.ndarray
    v_2: np.ndarray
    coords: np.ndarray
    h_local: np.ndarray
    free_index: int

    def kak_coords(self, theta: float) -> np.ndarray:
        """Return ``(a, b, c)`` at parameter value ``theta``."""
        return -theta * self.coords


def _hermitian_generator(matrix: np.ndarray, theta: float) -> np.ndarray:
    r"""Return the Hermitian ``H`` with ``matrix = exp(-i * theta * H)``.

    ``scipy`` is not a declared dependency, so the logarithm is taken by
    eigendecomposition, with ``np.angle`` selecting the principal branch.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    log_matrix = (
        eigenvectors @ np.diag(1j * np.angle(eigenvalues)) @ np.linalg.inv(eigenvectors)
    )
    generator = 1j * log_matrix / theta
    return (generator + generator.conj().T) / 2


def _split_generator(generator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a two-qubit Hermitian generator into its local and non-local parts.

    Returns ``(h_local, m)`` where ``h_local`` sums the terms carrying an identity
    factor and ``m[j - 1, k - 1]`` is the coefficient of ``sigma_j (x) sigma_k``.
    """
    h_local = np.zeros((4, 4), dtype=complex)
    m = np.zeros((3, 3))
    for j in range(4):
        for k in range(4):
            basis = np.kron(_PAULI[j], _PAULI[k])
            coefficient = np.trace(basis.conj().T @ generator) / 4
            if abs(coefficient) < _ATOL:
                continue
            if j == 0 or k == 0:
                h_local += coefficient * basis
            else:
                if abs(coefficient.imag) > _ATOL:  # pragma: no cover - defensive
                    raise QCutError(
                        f"non-local generator coefficient {coefficient} is not real, "
                        "so the gate is not a one-parameter group"
                    )
                m[j - 1, k - 1] = coefficient.real
    return h_local, m


def _so3_to_su2(rotation: np.ndarray) -> np.ndarray:
    """Lift ``rotation`` in SO(3) to the ``V`` in SU(2) that conjugates by it.

    Quaternion extraction taking the largest component first, since the naive
    ``w``-first formula degenerates at 180 degrees.
    """
    trace = np.trace(rotation)
    candidates = np.array(
        [
            1 + trace,
            1 + 2 * rotation[0, 0] - trace,
            1 + 2 * rotation[1, 1] - trace,
            1 + 2 * rotation[2, 2] - trace,
        ]
    )
    largest = int(np.argmax(candidates))
    value = np.sqrt(max(candidates[largest], 0.0)) / 2
    q = np.zeros(4)
    q[largest] = value
    if largest == 0:
        q[1] = (rotation[2, 1] - rotation[1, 2]) / (4 * value)
        q[2] = (rotation[0, 2] - rotation[2, 0]) / (4 * value)
        q[3] = (rotation[1, 0] - rotation[0, 1]) / (4 * value)
    else:
        i = largest - 1
        j, k = (i + 1) % 3, (i + 2) % 3
        q[0] = (rotation[k, j] - rotation[j, k]) / (4 * value)
        q[j + 1] = (rotation[j, i] + rotation[i, j]) / (4 * value)
        q[k + 1] = (rotation[k, i] + rotation[i, k]) / (4 * value)
    return q[0] * _I - 1j * (q[1] * _X + q[2] * _Y + q[3] * _Z)


def _canonicalise(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate the non-local coefficient matrix ``m`` to the canonical XX/YY/ZZ form.

    Returns ``(v_1, v_2, d)`` such that conjugating by ``v_1 (x) v_2`` turns
    ``sum_jk m[j, k] sigma_j (x) sigma_k`` into ``sum_i d[i] sigma_i (x) sigma_i``.
    """
    left, singular_values, right = np.linalg.svd(m)
    d = singular_values.copy()
    # SVD only guarantees orthogonal factors, so a reflection is moved into d to leave
    # both factors as proper rotations that can be lifted to SU(2).
    if np.linalg.det(left) < 0:
        left[:, 2] *= -1
        d[2] *= -1
    if np.linalg.det(right) < 0:
        right[2, :] *= -1
        d[2] *= -1
    return _so3_to_su2(left.T), _so3_to_su2(right), d


def _free_parameter_index(gate: Gate) -> int:
    """Return the index of the gate's single unbound parameter."""
    unbound = [
        index
        for index, parameter in enumerate(gate.params)
        if getattr(parameter, "parameters", None)
    ]
    if len(unbound) != 1:
        raise QCutError(
            f"analytic KAK needs exactly one unbound parameter, gate '{gate.name}' has "
            f"{len(unbound)}"
        )
    return unbound[0]


def analytic_kak(gate: Gate) -> AnalyticKAK:
    """Derive an unfolded KAK decomposition for a gate with one unbound parameter.

    Any other parameters must already be bound. They are held at their current values
    and folded into the constant locals.

    Args:
        gate: a two-qubit gate with exactly one unbound parameter.

    Returns:
        An :class:`AnalyticKAK` valid for every value of that parameter.

    Raises:
        QCutError: the gate is not two-qubit, has the wrong number of unbound
            parameters, or is not a one-parameter group.
    """
    if gate.num_qubits != 2:
        raise QCutError(
            f"analytic KAK needs a two-qubit gate, got '{gate.name}' on "
            f"{gate.num_qubits} qubits"
        )
    free_index = _free_parameter_index(gate)

    def at(theta: float) -> np.ndarray:
        params = list(gate.params)
        params[free_index] = theta
        probe = gate.copy()
        probe.params = params
        return probe.to_matrix()

    try:
        reference = at(REFERENCE_ANGLE)
    except Exception as exc:
        raise QCutError(
            f"could not evaluate gate '{gate.name}' at a reference angle, so analytic "
            "KAK needs the remaining parameters to be bound"
        ) from exc

    generator = _hermitian_generator(reference, REFERENCE_ANGLE)
    h_local, m = _split_generator(generator)
    non_local = sum(
        m[j, k] * np.kron(_PAULI[j + 1], _PAULI[k + 1])
        for j in range(3)
        for k in range(3)
    )
    if not isinstance(non_local, np.ndarray):
        raise QCutError(
            f"gate '{gate.name}' has no non-local generator, so it is a product of "
            "single-qubit gates and does not need cutting"
        )
    if np.abs(h_local @ non_local - non_local @ h_local).max() > _ATOL:
        raise QCutError(
            f"the local and non-local parts of gate '{gate.name}' do not commute, "
            "so it has no unfolded KAK form. Bind its parameters and cut it "
            "numerically."
        )

    v_1, v_2, coords = _canonicalise(m)
    decomposition = AnalyticKAK(v_1, v_2, coords, h_local, free_index)
    _verify(gate, at, decomposition)
    logger.debug(
        "analytic KAK for '%s', coords/theta=%s, local part %s",
        gate.name,
        np.array2string(coords, precision=4),
        "present" if np.abs(h_local).max() > _ATOL else "trivial",
    )
    return decomposition


#: Angles the derivation is checked at, straddling the Weyl-chamber fold where a
#: decomposition built from TwoQubitWeylDecomposition would break down.
_VERIFICATION_ANGLES = (0.2, 0.9, 1.7, 3.0, 4.4, 6.1)


def reassemble(decomposition: AnalyticKAK, theta: float) -> np.ndarray:
    """Rebuild the gate matrix from an :class:`AnalyticKAK`, up to global phase."""
    a, b, c = decomposition.kak_coords(theta)
    interaction = a * np.kron(_X, _X) + b * np.kron(_Y, _Y) + c * np.kron(_Z, _Z)
    basis = np.kron(decomposition.v_1, decomposition.v_2)
    core = basis.conj().T @ _expm_hermitian(interaction) @ basis
    return _expm_hermitian(-theta * decomposition.h_local) @ core


def _expm_hermitian(hermitian: np.ndarray) -> np.ndarray:
    """Return ``exp(i * hermitian)`` for a Hermitian matrix, via ``eigh``."""
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    return eigenvectors @ np.diag(np.exp(1j * eigenvalues)) @ eigenvectors.conj().T


def _verify(gate: Gate, at, decomposition: AnalyticKAK) -> None:
    """Check the decomposition reproduces the gate's channel across the range.

    Six 4x4 exponentials, worth doing eagerly because a silently wrong decomposition
    would otherwise surface only as biased expectation values much later.
    """
    for theta in _VERIFICATION_ANGLES:
        target = at(theta)
        rebuilt = reassemble(decomposition, theta)
        error = np.abs(
            np.kron(rebuilt, rebuilt.conj()) - np.kron(target, target.conj())
        ).max()
        if error > 1e-9:
            raise QCutError(
                f"analytic KAK for '{gate.name}' does not reproduce the gate at "
                f"theta={theta} with channel error {error:.2e}, so it is probably not "
                "a one-parameter group. Bind its parameters and cut it numerically."
            )
