"""Group cut locations that share a single quasiprobability decomposition.

Cutting each location separately costs the product of their sampling overheads. Some
groups of cuts have a joint decomposition that costs strictly less, so a QPD term has to
be able to span several cut locations at once.

A :class:`Bundle` names the locations that share one decomposition. Its terms carry an
operation per *side* acting on as many qubits as the bundle holds, where side 0 is the
control-side placeholder of every cut and side 1 the target-side one. A bundle of one
is exactly the single-cut case, with one-qubit operations, and takes the original code
path.

The only joint decomposition here is :mod:`QCut.qpd_joint`, for parallel two-qubit
rotation gates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate

from QCut.cutlocation import CutLocation
from QCut.options import CutOptions
from QCut.qpd_joint import gamma_joint, gamma_separate, single_axis_frame

logger: logging.Logger = logging.getLogger(__name__)

#: Placeholder side indices. Side 0 holds the ``_c_`` and ``Meas`` placeholders, side 1
#: the ``_t_`` and ``Init`` ones.
SIDE_0, SIDE_1 = 0, 1


@dataclass(frozen=True)
class Bundle:
    """Cut locations sharing one quasiprobability decomposition.

    ``layout[b][s]`` names the ``(cut, placeholder side)`` that bundle side ``b`` puts
    on its qubit ``s``. Each bundle side lives entirely in one subcircuit, which is what
    lets its operation act on several qubits at once. A cut whose placeholder sides run
    the other way round to its neighbours still joins the bundle, on the far side, since
    a single-axis gate is symmetric about its interaction axis.

    ``place[b]`` says whether side ``b``'s operation goes in where the first or the last
    of its placeholders sat. Which one works depends on the circuit, see
    :func:`_placement`.
    """

    cuts: tuple[int, ...]
    kind: str = "single"
    layout: tuple[tuple[tuple[int, int], ...], ...] = ()
    place: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        """How many cut locations the bundle covers."""
        return len(self.cuts)

    @property
    def anchor(self) -> int:
        """The cut that carries the bundle's operations once terms are flattened."""
        return self.cuts[0]

    @property
    def flipped(self) -> list[bool]:
        """Per cut, whether its two qubits are reversed relative to the bundle sides."""
        return [side == SIDE_1 for _, side in self.layout[0]]

    def bundle_side(self, cut: int, side: int) -> int | None:
        """Return which bundle side holds a placeholder, or None if it is not in one."""
        for index, members in enumerate(self.layout):
            if (cut, side) in members:
                return index
        return None


@dataclass(frozen=True)
class Placeholder:
    """Where one side of one cut ended up after the circuit was split."""

    subcircuit: int
    index: int
    qubit: int


def parse_placeholder(name: str) -> tuple[int, int] | None:
    """Return ``(cut_index, side)`` for a placeholder name, else None."""
    if "_c_" in name:
        side = SIDE_0
    elif "_t_" in name:
        side = SIDE_1
    elif name.startswith("Meas_"):
        side = SIDE_0
    elif name.startswith("Init_"):
        side = SIDE_1
    else:
        return None
    try:
        return int(name.split("_")[-1]), side
    except ValueError:  # pragma: no cover - defensive
        return None


def locate_placeholders(
    subcircuits: list[QuantumCircuit],
) -> dict[tuple[int, int], Placeholder]:
    """Map ``(cut index, side)`` to where that placeholder sits after the split."""
    found: dict[tuple[int, int], Placeholder] = {}
    for sub_index, subcircuit in enumerate(subcircuits):
        for index, instruction in enumerate(subcircuit.data):
            parsed = parse_placeholder(instruction.operation.name)
            if parsed is None:
                continue
            found[parsed] = Placeholder(
                sub_index, index, subcircuit.find_bit(instruction.qubits[0]).index
            )
    return found


def _placement(
    subcircuit: QuantumCircuit, placeholders: list[Placeholder]
) -> str | None:
    """Where one multi-qubit operation replacing ``placeholders`` can go, if anywhere.

    The block needs a single point in the subcircuit, so every placeholder has to be
    able to slide there without crossing anything on *its own* qubit. Operations on the
    group's other qubits are irrelevant, which is the point: the cut gates are parallel,
    so each slot slides independently.

    Sliding the whole group to the earliest placeholder and to the latest are different
    questions and either can be the one that works, so both are tried. Returns
    ``"first"``, ``"last"``, or None if the cuts are not parallel after all.
    """
    if len(placeholders) < 2:
        return "first"
    qubits = [placeholder.qubit for placeholder in placeholders]
    if len(set(qubits)) != len(qubits):
        # Two cuts sharing a qubit are not parallel and the block cannot use it twice.
        return None
    own = {placeholder.index for placeholder in placeholders}
    for name, target in (("first", min(own)), ("last", max(own))):
        if all(
            _clear_between(subcircuit, placeholder, target, own)
            for placeholder in placeholders
        ):
            return name
    return None


def _clear_between(
    subcircuit: QuantumCircuit, placeholder: Placeholder, target: int, own: set[int]
) -> bool:
    """Whether a placeholder can slide to ``target`` without crossing its own wire."""
    for index in range(min(placeholder.index, target), max(placeholder.index, target)):
        if index in own:
            continue
        instruction = subcircuit.data[index]
        if placeholder.qubit in {
            subcircuit.find_bit(qubit).index for qubit in instruction.qubits
        }:
            return False
    return True


def _candidate_groups(
    cut_locations: list, placeholders: dict[tuple[int, int], Placeholder]
) -> dict[tuple[int, int], list[int]]:
    """Group gate cuts by which two subcircuits they join, ignoring which way round."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, location in enumerate(cut_locations):
        if not isinstance(location, CutLocation):
            continue
        side_0 = placeholders.get((index, SIDE_0))
        side_1 = placeholders.get((index, SIDE_1))
        if side_0 is None or side_1 is None:
            continue
        if side_0.subcircuit == side_1.subcircuit:  # pragma: no cover - defensive
            continue
        # A single-axis gate is symmetric about its interaction axis, so a cut whose
        # sides are the other way round joins the same bundle with its locals swapped.
        key = (
            (side_0.subcircuit, side_1.subcircuit)
            if side_0.subcircuit < side_1.subcircuit
            else (side_1.subcircuit, side_0.subcircuit)
        )
        groups.setdefault(key, []).append(index)
    return groups


def is_joint_eligible(location) -> bool:
    """Whether a cut location's gate has a single non-zero interaction coordinate."""
    if not isinstance(location, CutLocation):
        return False
    gate = joint_gate(location)
    return gate is not None and single_axis_frame(gate) is not None


def joint_gate(location: CutLocation):
    """Return the gate a joint decomposition would be built from, or None.

    The hand-written tables cover ``cz``, ``swap`` and ``iswap``. Only ``cz`` is
    single-axis, and it needs a gate object to derive its frame from, so it is
    reconstructed here when the marker did not carry one.
    """
    if location.gate is not None:
        return location.gate
    if location.gate_name == "cz":
        return CZGate()
    return None


def _group_placeholders(
    placeholders: dict[tuple[int, int], Placeholder],
    cuts: list[int],
    subcircuit: int,
) -> list[Placeholder]:
    """Return the group's placeholders that landed in one subcircuit."""
    return [
        placeholders[(cut, side)]
        for cut in cuts
        for side in (SIDE_0, SIDE_1)
        if placeholders[(cut, side)].subcircuit == subcircuit
    ]


def _grow_groups(
    eligible: list[int],
    placeholders: dict[tuple[int, int], Placeholder],
    subcircuits: list[QuantumCircuit],
    pair: tuple[int, int],
) -> list[tuple[list[int], tuple[str, str]]]:
    """Split ``eligible`` into groups that one operation per side can replace.

    Grown one cut at a time rather than solved exactly. A cut that cannot join the
    current group starts a new one, which keeps the result deterministic and is enough
    for the layered circuits joint cutting is aimed at.
    """

    def placements(group: list[int]) -> tuple[str, str] | None:
        found = []
        for sub in pair:
            where = _placement(
                subcircuits[sub], _group_placeholders(placeholders, group, sub)
            )
            if where is None:
                return None
            found.append(where)
        return (found[0], found[1])

    groups: list[tuple[list[int], tuple[str, str]]] = []
    current: list[int] = []
    current_place: tuple[str, str] | None = None
    for cut in eligible:
        candidate = placements(current + [cut])
        if candidate is not None:
            current = current + [cut]
            current_place = candidate
            continue
        if len(current) > 1 and current_place is not None:
            groups.append((current, current_place))
        current = [cut]
        current_place = placements(current)
    if len(current) > 1 and current_place is not None:
        groups.append((current, current_place))
    return groups


def _layout(
    group: list[int],
    placeholders: dict[tuple[int, int], Placeholder],
    pair: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Assign each cut's two placeholders to a bundle side.

    Bundle side 0 is the lower-numbered subcircuit, so a bundle side always sits wholly
    inside one subcircuit and its operation can span all of that side's qubits.
    """
    sides: list[list[tuple[int, int]]] = [[], []]
    for cut in group:
        for side in (SIDE_0, SIDE_1):
            which = 0 if placeholders[(cut, side)].subcircuit == pair[0] else 1
            sides[which].append((cut, side))
    return tuple(tuple(members) for members in sides)


def plan_bundles(
    cut_locations: list,
    subcircuits: list[QuantumCircuit],
    options: CutOptions,
    announce: bool = True,
) -> list[Bundle]:
    """Decide which cut locations share a decomposition.

    Every cut ends up in exactly one bundle, and the bundles come back ordered by their
    first cut so the experiment tensor is reproducible.

    Args:
        cut_locations: the cuts to group.
        subcircuits: the split circuit, needed to tell which cuts are parallel and which
            subcircuits each cut joins.
        options: configuration. ``joint_rotation_cuts`` turns joint cutting off.
        announce: whether to log what was bundled. Off while costing a candidate plan,
            which would otherwise report a grouping that may not be the one used.

    Returns:
        One :class:`Bundle` per group, of kind ``"joint_rotation"`` where a joint
        decomposition applies and ``"single"`` everywhere else.
    """
    bundled: dict[int, Bundle] = {}
    if options.joint_rotation_cuts:
        placeholders = locate_placeholders(subcircuits)
        candidates = _candidate_groups(cut_locations, placeholders)
        for pair, members in candidates.items():
            eligible = [
                index for index in members if is_joint_eligible(cut_locations[index])
            ]
            for group, place in _grow_groups(eligible, placeholders, subcircuits, pair):
                bundle = Bundle(
                    tuple(group),
                    "joint_rotation",
                    _layout(group, placeholders, pair),
                    place,
                )
                for index in group:
                    bundled[index] = bundle

    bundles: list[Bundle] = []
    seen: set[Bundle] = set()
    for index in range(len(cut_locations)):
        bundle = bundled.get(index, Bundle((index,), "single"))
        if bundle in seen:
            continue
        seen.add(bundle)
        bundles.append(bundle)

    if announce:
        _log_savings(bundles, cut_locations)
    return bundles


def _log_savings(bundles: list[Bundle], cut_locations: list) -> None:
    """Report what bundling bought, since it changes both gamma and the group count."""
    joint = [bundle for bundle in bundles if bundle.size > 1]
    if not joint:
        return
    together, apart = 1.0, 1.0
    for bundle in joint:
        thetas = [
            single_axis_frame(joint_gate(cut_locations[index]))[0]
            for index in bundle.cuts
        ]
        together *= gamma_joint(thetas)
        apart *= gamma_separate(thetas)
    logger.info(
        "Bundled %d cut(s) into %d joint decomposition(s), gamma %.4f against %.4f "
        "cut separately.",
        sum(bundle.size for bundle in joint),
        len(joint),
        together,
        apart,
    )


def flatten_term(bundle: Bundle, term: dict) -> dict[int, dict]:
    """Spread one bundle term over its cut locations.

    The anchor keeps the operations and the whole coefficient, and the rest get a
    coefficient of one and no operations, meaning their placeholder is covered by the
    anchor's and should just be removed. Everything downstream that multiplies the
    per-cut coefficients together therefore keeps working unchanged.

    A bundle of one hands back the term itself rather than a copy, so single cuts behave
    exactly as they did before bundling existed.
    """
    if bundle.size == 1:
        return {bundle.anchor: term}
    spread = {bundle.anchor: dict(term)}
    for index in bundle.cuts[1:]:
        spread[index] = {"op_0": None, "op_1": None, "c": 1.0}
    return spread
