"""
Utility functions for running on real backends.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from qiskit import transpile
from qiskit.circuit import Gate
from qiskit.transpiler import Target

from QCut.bundle import locate_placeholders, plan_bundles
from QCut.circuit_utils import (
    MarkerSpan,
    _drop_barriers,
    _fence_markers,
    _record_layout,
    barriers_to_markers,
    fuse_markers,
    markers_to_barriers,
    split_markers,
)
from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation, SingleQubitCutLocation

logger: logging.Logger = logging.getLogger(__name__)


#: How wide a block one marker will stand in for. Two covers the case that pays: a pair
#: of wire cuts communicating costs gamma 7 against the 16 of cutting them locally.
#: Wider blocks need all of their wires mutually coupled, which no device this targets
#: has, so they are left to fall back to narrower ones.
WIDEST_BLOCK_MARKER: int = 2

#: Prefix of the marker that stands in for a whole block while transpiling.
BLOCK_MARKER: str = "qcut_block"


def _block_spans(
    cut_circuit,
) -> tuple[dict[int, list[MarkerSpan]], dict[str, list[str]]]:
    """Find the placeholder runs that should be transpiled as one wider marker.

    Planning happens here, on the logical subcircuits, because the whole point is to
    tell the transpiler something before it runs. The experiment builder plans again
    afterwards and vetoes any block the device does not couple, but a block routed as
    one marker passes that veto, and its placeholders come back consecutive, so it is
    found again.
    """
    bundles = plan_bundles(
        cut_circuit.cut_locations,
        cut_circuit.subcircuits,
        cut_circuit.options,
        announce=False,
        fits=lambda bundle: bundle.size <= WIDEST_BLOCK_MARKER,
    )
    placeholders = locate_placeholders(cut_circuit.subcircuits)
    spans: dict[int, list[MarkerSpan]] = defaultdict(list)
    members: dict[str, list[str]] = {}
    for order, bundle in enumerate(bundles):
        if bundle.size < 2:
            continue
        for side, group in enumerate(bundle.layout):
            owner = placeholders[group[0]].subcircuit
            subcircuit = cut_circuit.subcircuits[owner]
            indices = [placeholders[member].index for member in group]
            name = f"{BLOCK_MARKER}_{order}_{side}"
            spans[owner].append(
                MarkerSpan(
                    name,
                    tuple(placeholders[member].qubit for member in group),
                    tuple(indices),
                    min(indices) if bundle.place[side] == "first" else max(indices),
                )
            )
            members[name] = [subcircuit.data[index].operation.name for index in indices]
    return spans, members


def _add_block_gates(target, coupling_map, members: dict[str, list[str]]) -> set[str]:
    """Declare each block marker on both directions of every coupled pair.

    Both directions, because a device's coupling map names each pair once and the layout
    is free to hand the marker its qubits either way round. Qiskit can turn a ``cz`` the
    right way because it knows the gate is symmetric; a marker is opaque, so an
    unsupported orientation is a transpiler error rather than something it can fix.
    """
    loci = {}
    for first, second in coupling_map.get_edges():
        loci[(first, second)] = None
        loci[(second, first)] = None
    for name, group in members.items():
        target.add_instruction(
            Gate(num_qubits=len(group), name=name, params=[], label=name),
            loci,
            name=name,
        )
    return set(members)


def _markers_in(circuit, names: set[str]) -> list[str]:
    """The placeholders a circuit carries, sorted, for comparing before with after."""
    return sorted(
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name in names
    )


def _translate_subcircuit(subcircuit, spans, translate, marker_names):
    """Transpile one subcircuit with its blocks fused, or plainly if that loses a cut.

    Standing a wider marker in for a block is an optimisation: it costs at most a swap
    and saves far more in shots. Losing a placeholder to it would not be a worse
    optimisation but a wrong answer, and one nothing downstream could notice -- the cut
    would simply not be there. So the fused attempt is checked against the placeholders
    that went in, and anything short of all of them coming back falls back.
    """
    if not spans:
        return translate(subcircuit, [])

    expected = _markers_in(subcircuit, marker_names)
    try:
        fused = translate(subcircuit, spans)
    except Exception as error:  # noqa: BLE001 - a failed optimisation must not propagate
        logger.info("could not transpile this subcircuit's blocks as one: %s", error)
        return translate(subcircuit, [])

    if _markers_in(fused, marker_names) == expected:
        return fused

    logger.info(
        "the wider marker standing in for a block did not survive transpilation, so "
        "this subcircuit is transpiled without it. Its cuts are then decomposed "
        "separately, which costs more shots but is what the device can run."
    )
    return translate(subcircuit, [])


def _placeholder_gates(cut_circuit) -> dict[str, Gate]:
    """One opaque one-qubit gate per placeholder the subcircuits carry.

    The transpiler is told about them through the target, so it leaves them alone
    instead of refusing to synthesise something it has never heard of.
    """
    gates: dict[str, Gate] = {}

    def add(name: str, width: int = 1) -> None:
        gates[name] = Gate(num_qubits=width, name=name, params=[], label=name)

    for ind, location in enumerate(cut_circuit.cut_locations):
        if isinstance(location, CutLocation):
            add(f"cut{location.gate_name.upper()}_t_{ind}")
            add(f"cut{location.gate_name.upper()}_c_{ind}")
        elif isinstance(location, SingleQubitCutLocation):
            add(f"Meas_{ind}")
            add(f"Init_{ind}")

    for index in range(sum(x.num_qubits for x in cut_circuit.subcircuits)):
        add(f"obs_{index}")
    return gates


def _iqm_transpiler_for(backend):
    """Return IQM's transpiler if it is installed and ``backend`` is one of its own.

    The adapter comes with ``pip install "QCut[iqm]"``. Without it, or for any other
    backend, this returns None and the ordinary qiskit path is used. The backend check
    is what ``IQMBackendBase`` is imported for: a fake IQM device and a real one both
    derive from it, and nothing else does.
    """
    try:
        from iqm.qiskit_iqm import transpile_to_IQM
        from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
    except ImportError:
        return None
    return transpile_to_IQM if isinstance(backend, IQMBackendBase) else None


def transpile_subcircuits(
    cut_circuit: CutCircuit,
    backend,
    optimization_level: int = 3,
    transpile_options: dict | None = None,
    use_iqm_transpiler: bool = True,
) -> CutCircuit:
    """
    Transpile subcircuits for a given backend. More efficient than transpiling
    experiment circuits as it only transpiles each subcircuit once instead of
    each experiment circuit. However, may lead to suboptimal transpilation results as
    the tranpiler cannot use the backend object directly due to need to retain
    some placeholder gates for cuts and observables. `transpile_options` can be used
    to pass additional options to the transpiler. For more control over transpilation
    of experiment circuits, use `transpile_experiments` or manually transpile them.

    Args:
        subcircuits (list[QuantumCircuit]): List of subcircuits to be transpiled.
        backend: Backend to transpile to.
        optimization_level (int): Optimization level for transpilation (0-3).
        transpile_options (dict): Arguments passed to qiskit transpile function.
    Returns:
        CutCircuit: Transpiled subcircuits wrapped in CutCircuit class.
    """

    if not isinstance(cut_circuit, CutCircuit):
        raise ValueError("cut_circuit must be of type CutCircuit.")

    custom_gates = _placeholder_gates(cut_circuit)

    marker_names = set(custom_gates)
    # Where several placeholders will be replaced by one block, so it can be routed as
    # the multi-qubit operation it is rather than as unrelated single-qubit ones.
    spans, block_members = _block_spans(cut_circuit)

    target = Target()

    try:
        basis_gates = list({i[0].name for i in backend._target.instructions})

    except Exception as e:
        raise ValueError(f"Error accessing backend target instructions: {e}")

    target = target.from_configuration(
        num_qubits=backend.num_qubits,
        coupling_map=backend._coupling_map,
        basis_gates=basis_gates + list(custom_gates.keys()),
        custom_name_mapping=custom_gates,
    )
    block_names = _add_block_gates(target, backend._coupling_map, block_members)

    # Fenced first: a placeholder is a single-qubit gate as far as the transpiler is
    # concerned, so without barriers it gets commuted past the others and the experiment
    # builder, which reads them in order, misreads the result.

    iqm_transpile = _iqm_transpiler_for(backend) if use_iqm_transpiler else None
    if iqm_transpile is not None:
        # IQM's own transpiler produces markedly shallower circuits than the generic
        # path, and it will accept the placeholders once they are barriers carrying
        # their name. Two of its defaults have to go the other way for QCut:
        #
        #   remove_final_rzs   a Z rotation before a Z measurement is unobservable, so
        #                      it drops trailing ones. QCut adds the rotations for X and
        #                      Y observables later, and those frames are needed then.
        #   perform_move_routing
        #                      rebuilds the classical registers on the way to the Star
        #                      architecture and loses ``qpd_meas`` wherever a term does
        #                      not write to it. Inserting the MOVEs belongs at
        #                      submission, which is where iqm-client does it.
        #
        # Both are still overridable through ``transpile_options``, deliberately, but
        # the answers will be wrong.
        options = {
            "remove_final_rzs": False,
            "perform_move_routing": False,
            "optimization_level": optimization_level,
        }
        options.update(transpile_options or {})

        # A custom gate is refused here, so the block marker has to be a native one
        # carrying a label -- which this transpiler, unlike qiskit's, does preserve. It
        # is fenced because a real gate would otherwise be a real licence to commute
        # operations through it, and the block replacing it grants no such licence.
        # Nothing is fused here: a custom gate is refused by this transpiler, so a block
        # marker would have to become a barrier, and a barrier constrains no routing.
        # The blocks fall back to narrower ones, which need only one-qubit operations.
        transpiled = [
            barriers_to_markers(
                _record_layout(
                    iqm_transpile(
                        markers_to_barriers(subcircuit, marker_names),
                        backend,
                        **options,
                    ),
                    subcircuit.num_qubits,
                ),
                marker_names,
            )
            for subcircuit in cut_circuit.subcircuits
        ]
        return CutCircuit(
            subcircuits=transpiled,
            cut_locations=cut_circuit.cut_locations,
            map_qubit=cut_circuit.map_qubit,
            options=cut_circuit.options,
            backend=backend,
        )

    def qiskit_translate(subcircuit, spans_for):
        translated = transpile(
            _fence_markers(
                fuse_markers(
                    subcircuit,
                    spans_for,
                    lambda span: target.operation_from_name(span.name),
                ),
                marker_names | block_names,
            ),
            target=target,
            optimization_level=optimization_level,
            **(transpile_options or {}),
        )
        # Undo the layout: transpiling lays the subcircuits out on physical qubits and
        # pads them to the device width, and post-processing reads measurement bits by
        # position.
        return split_markers(
            _drop_barriers(_record_layout(translated, subcircuit.num_qubits)),
            block_members,
        )

    transpiled = [
        _translate_subcircuit(
            subcircuit, spans.get(index, []), qiskit_translate, marker_names
        )
        for index, subcircuit in enumerate(cut_circuit.subcircuits)
    ]

    return CutCircuit(
        subcircuits=transpiled,
        cut_locations=cut_circuit.cut_locations,
        map_qubit=cut_circuit.map_qubit,
        options=cut_circuit.options,
        backend=backend,
    )


def transpile_experiments(
    cut_experiment: CutExperiment,
    backend,
    optimization_level: int = 0,
    transpile_options: dict | None = None,
    use_iqm_transpiler: bool = True,
) -> CutExperiment:
    """
    Transpile experiment circuits. Transpiles all generated experiment circuits for
    a given backend. Most often one should use `transpile_subcircuits` instead, as that
    only transpiles subcircuits before experiment generation which is a lot more
    efficient. This function is mainly provided for special cases where one needs/wants
    extra control over the transpilation of experiment circuits.

    As with :func:`transpile_subcircuits`, an IQM backend is handed to IQM's own
    transpiler when the adapter is installed, with the same two defaults inverted and
    for the same reasons. No placeholders are left at this point, so nothing has to be
    hidden from it as barriers, but the layout still has to be undone afterwards.

    Args:
        cut_experiment: (CutExperiment): Experiment circuits to be transpiled.
        backend (str): Backend to transpile to.
        optimization_level (int): Optimization level for transpilation (0-3).
        transpile_options (dict): Arguments passed to the transpiler.
        use_iqm_transpiler (bool): Whether an IQM backend may use IQM's transpiler.
            Pass False for the ordinary qiskit path.

    Returns:
        CutExperiment: Transpiled experiment circuits wrapped in CutExperiment class.
    """

    if not isinstance(cut_experiment, CutExperiment):
        raise ValueError("cut_experiment must be of type CutExperiment.")

    iqm_transpile = _iqm_transpiler_for(backend) if use_iqm_transpiler else None

    if iqm_transpile is not None:
        options = {
            "remove_final_rzs": False,
            "perform_move_routing": False,
            "optimization_level": optimization_level,
        }
        options.update(transpile_options or {})

        def translate(circuit):
            return _record_layout(
                iqm_transpile(circuit, backend, **options), circuit.num_qubits
            )
    else:
        # Not ``backend=backend``: a resonator machine's target carries a ``move``
        # operation, and transpiling against it emits MOVE gates that no local
        # simulator can run and whose register rewriting QCut cannot reconstruct
        # from. Building the target from the backend's instruction names leaves that
        # operation out, and the resonator stage happens at submission instead.
        basis = sorted({item[0].name for item in backend._target.instructions})
        fallback_target = Target().from_configuration(
            num_qubits=backend.num_qubits,
            coupling_map=backend._coupling_map,
            basis_gates=basis,
        )

        def translate(circuit):
            return _record_layout(
                transpile(
                    circuit,
                    target=fallback_target,
                    optimization_level=optimization_level,
                    **(transpile_options or {}),
                ),
                circuit.num_qubits,
            )

    subexperiments = [
        [{ind: translate(circ) for ind, circ in exp.items()} for exp in exps]
        for exps in cut_experiment.experiments
    ]

    return CutExperiment(
        subexperiments,
        cut_locations=cut_experiment.cut_locations,
        map_qubit=cut_experiment.map_qubit,
        coefficients=cut_experiment.coefficients,
        observables=cut_experiment.observables,
        options=cut_experiment.options,
        backend=backend,
        # Translating gates does not change any of these, and dropping them would.
        # Without the plan a communicating experiment forgets that it runs in waves and
        # is executed as though nothing depended on a measured outcome; without the bit
        # layout the estimator cannot find the qpd bits.
        num_draws=cut_experiment._num_draws,
        plan=cut_experiment.plan,
        qpd_bits=cut_experiment.qpd_bits,
    )
