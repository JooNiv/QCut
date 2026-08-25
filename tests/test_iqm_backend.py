"""End-to-end checks against a backend with a coupling map.

Nothing else in the suite runs a cut experiment against a backend that has a real
topology, and that gap hid a run of defects: transpilation lays subcircuits out on
physical qubits and pads them to the device width, so reading measurement bits by
position attributed results to the wrong qubits; placeholders were looked up by the
index they had before transpiling; routing borrowed wires that were then measured; a
resonator machine's gate list broke the basis translation; and unused classical
registers made real hardware refuse the job outright. Every one of those passed
``pytest`` while returning wrong numbers or none at all, because the only backend
test asserted gate names.

So these assert values, and that the circuits would be accepted.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutGate
from QCut.bundle import plan_bundles
from QCut.qpd_operations import coupling_filter

iqm = pytest.importorskip(
    "iqm.qiskit_iqm", reason='needs the IQM adapter: pip install "QCut[iqm]"'
)

#: Loose, because these run few shots on purpose: the failures being guarded against are
#: wrong-qubit and wrong-register ones, which miss by order one, not by a fraction.
TOLERANCE = 0.15
SHOTS = 2**12


def _backends():
    return [
        pytest.param(iqm.IQMFakeAdonis(), id="adonis_star"),
        pytest.param(iqm.IQMFakeDeneb(), id="deneb_resonator"),
        pytest.param(iqm.IQMFakeApollo(), id="apollo"),
    ]


def _direct_backends():
    """The backends that couple qubits to each other, so a plain two-qubit gate runs.

    A resonator machine is excluded on purpose: it couples qubits only through its
    resonator, so it refuses any two-qubit gate until the MOVEs are inserted at
    submission, whatever the layout.
    :func:`test_a_resonator_device_needs_its_moves_inserted` covers that separately.
    """
    return [
        pytest.param(iqm.IQMFakeAdonis(), id="adonis_star"),
        pytest.param(iqm.IQMFakeApollo(), id="apollo"),
    ]


def _gate_cut():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.3 + 0.2 * qubit, qubit)
        circuit.rzz(0.4, 0, 1)
        circuit.rzz(0.5, 2, 3)
    marked.append(**cutGate(RZZGate(0.9), 1, 2))
    plain.rzz(0.9, 1, 2)
    return marked, plain, SparsePauliOp(["IIIZ", "IIZI", "IZII", "ZIII"])


def _wire_cut():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.3 + 0.2 * qubit, qubit)
        circuit.cx(0, 1)
    marked.append(cut(), [1])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.cx(2, 3)
    return marked, plain, SparsePauliOp(["IIIZ", "IIZI", "IZII", "ZIII"])


def _locc_block():
    marked, plain = QuantumCircuit(6), QuantumCircuit(6)
    for circuit in (marked, plain):
        for qubit in range(6):
            circuit.ry(0.3 + 0.13 * qubit, qubit)
        circuit.cx(0, 1)
    marked.append(cut(), [0])
    marked.append(cut(), [1])
    for circuit in (marked, plain):
        circuit.cx(0, 2)
        circuit.cx(1, 3)
    return marked, plain, SparsePauliOp(["IIIIIZ", "IIIIZI"])


def _locc_pair():
    """Two cuts that really do share one block, unlike :func:`_locc_block`.

    That one bundles each wire on its own, so its terms are one-qubit operations and it
    never exercised the case this file exists for: a block spanning two wires puts a
    two-qubit gate in after transpilation, at whatever wires the layout happened to give
    the cuts, with nothing routing it.
    """
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
    marked.append(cut(), [1])
    marked.append(cut(), [2])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.cx(2, 3)
    return marked, plain, SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])


CASES = {
    "gate_cut": _gate_cut,
    "wire_cut": _wire_cut,
    "locc_block": _locc_block,
    "locc_pair": _locc_pair,
}


def _spans_two_wires(bundles):
    return any(bundle.size > 1 for bundle in bundles)


def _every_register_is_written(circuit):
    """IQMBackend.run refuses a job whose circuit has a register nothing writes to."""
    utilisation = dict.fromkeys(circuit.cregs, 0)
    for instruction in circuit.data:
        if instruction.clbits:
            register = circuit.find_bit(instruction.clbits[0]).registers[0][0]
            utilisation[register] += 1
    return 0 not in utilisation.values()


def _accepted_by(backend, experiment, shots=32):
    """Submit to the device itself, which is the only thing that validates the topology.

    Running the experiment on an ideal simulator checks the arithmetic, and Aer will
    happily execute a two-qubit gate on any pair of wires. Only the backend checks that
    each gate sits on a locus the architecture actually has, which is what caught the
    layout being undone rather than honoured: the answers were right and the circuits
    were unrunnable.
    """
    ck.run_experiments(experiment, backend=backend, shots=shots)


def _run(marked, observables, backend, level, use_iqm):
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    transpiled = ck.transpile_subcircuits(
        cut_circuit, backend, optimization_level=level, use_iqm_transpiler=use_iqm
    )
    experiment = ck.get_experiment_circuits(transpiled, observables)
    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    return experiment, np.array(ck.estimate_expectation_values(results))


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize("use_iqm", [True, False], ids=["iqm_transpiler", "qiskit"])
def test_a_cut_experiment_survives_a_real_topology(backend, case, level, use_iqm):
    """The answer has to come back right after being laid out on physical qubits."""
    marked, plain, observables = CASES[case]()
    state = Statevector(plain)
    exact = np.array(
        [float(np.real(state.expectation_value(p))) for p in observables.paulis]
    )

    _, values = _run(marked, observables, backend, level, use_iqm)

    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE, (
            f"{case} on {backend.name} at O{level}: expected {expected}, got {actual}"
        )


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_no_circuit_carries_an_unwritten_register(case):
    """Real hardware refuses those, and QCut used to emit them for most groups."""
    marked, _plain, observables = CASES[case]()
    experiment, _ = _run(marked, observables, iqm.IQMFakeAdonis(), 3, True)

    offenders = [
        (group, obs, sub)
        for group, groups in enumerate(experiment.experiments)
        for obs, obs_group in enumerate(groups)
        for sub, circuit in obs_group.items()
        if not _every_register_is_written(circuit)
    ]
    assert offenders == []


@pytest.mark.sim
@pytest.mark.slow
def test_the_subcircuits_end_up_in_the_backend_basis():
    """Whichever transpiler ran, the result has to be something the device can run."""
    marked, _plain, observables = _gate_cut()
    for use_iqm in (True, False):
        cut_circuit = ck.get_locations_and_subcircuits(marked.copy())
        transpiled = ck.transpile_subcircuits(
            cut_circuit,
            iqm.IQMFakeDeneb(),
            optimization_level=3,
            use_iqm_transpiler=use_iqm,
        )
        for subcircuit in transpiled.subcircuits:
            for instruction in subcircuit.data:
                name = instruction.operation.name
                assert name in ("r", "cz", "barrier", "measure") or name.startswith(
                    ("cut", "obs_", "Meas_", "Init_")
                ), f"unexpected {name} with use_iqm_transpiler={use_iqm}"


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    _direct_backends(),
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_device_accepts_the_circuits(backend, case):
    """Every gate has to sit on a locus the architecture has.

    Transpiling chooses a layout so the two-qubit gates land on coupled pairs. Renaming
    the wires afterwards to put the subcircuit's own qubits back in order would throw
    that away and the backend would refuse the job, so the placement is kept and the
    measurements go through the recorded layout instead.
    """
    marked, _plain, observables = CASES[case]()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)
    experiment = ck.get_experiment_circuits(transpiled, observables)

    _accepted_by(backend, experiment)


@pytest.mark.sim
@pytest.mark.slow
def test_a_resonator_device_needs_its_moves_inserted():
    """A star machine couples qubits only through its resonator.

    So a circuit carrying a plain two-qubit gate is not runnable there whatever the
    layout, and QCut deliberately stops at the simplified architecture: inserting the
    MOVEs belongs at submission, and a move-routed circuit cannot be simulated locally
    at all. This pins both halves -- that the device refuses it before, and that
    inserting the moves is enough, with the classical registers surviving.
    """
    backend = iqm.IQMFakeDeneb()
    marked, _plain, observables = _gate_cut()
    cut_circuit = ck.get_locations_and_subcircuits(marked.copy())
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)

    before = ck.get_experiment_circuits(transpiled, observables)
    with pytest.raises(Exception, match="not allowed as locus"):
        _accepted_by(backend, before)

    after = ck.get_experiment_circuits(transpiled, observables)
    for groups in after.experiments:
        for obs_group in groups:
            for sub, circuit in list(obs_group.items()):
                shapes = [(r.name, r.size) for r in circuit.cregs]
                routed = iqm.transpile_to_IQM(
                    circuit,
                    backend,
                    remove_final_rzs=False,
                    perform_move_routing=True,
                    optimization_level=0,
                )
                assert [(r.name, r.size) for r in routed.cregs] == shapes
                obs_group[sub] = routed

    _accepted_by(backend, after)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
@pytest.mark.parametrize("use_iqm", [True, False], ids=["iqm_transpiler", "qiskit"])
def test_a_block_spanning_two_wires_is_not_left_unrouted(backend, use_iqm):
    """A bundle's two-qubit gates have to sit on a locus the device has.

    They go in after transpilation, so the transpiler never saw them and had no reason
    to put the cuts' wires next to each other -- on a star machine it usually cannot.
    Planning has to notice and fall back to cutting the wires separately, which needs
    only one-qubit operations.
    """
    marked, _plain, observables = _locc_pair()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    assert _spans_two_wires(
        plan_bundles(
            cut_circuit.cut_locations, cut_circuit.subcircuits, cut_circuit.options
        )
    ), "the case has stopped exercising a multi-wire block"

    transpiled = ck.transpile_subcircuits(
        cut_circuit, backend, optimization_level=3, use_iqm_transpiler=use_iqm
    )
    experiment = ck.get_experiment_circuits(transpiled, observables)
    _accepted_by(backend, experiment)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
def test_transpiling_the_experiments_keeps_the_joint_decomposition(backend):
    """The fallback is only needed because the block goes in after transpilation.

    Build the experiment circuits first and the block is routed with everything else, so
    the cheaper decomposition survives and the device still takes the job.
    """
    marked, plain, observables = _locc_pair()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    separate = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(
            marked.copy(), options=CutOptions(wire_cut_communication="never")
        ),
        observables,
    )
    assert experiment.num_circuits < separate.num_circuits

    transpiled = ck.transpile_experiments(experiment, backend, optimization_level=3)
    assert transpiled.num_circuits == experiment.num_circuits
    _accepted_by(backend, transpiled)

    results = ck.run_experiments(transpiled, backend=AerSimulator(), shots=SHOTS)
    values = np.array(ck.estimate_expectation_values(results))
    exact = np.array(
        [
            float(np.real(Statevector(plain).expectation_value(pauli)))
            for pauli in observables.paulis
        ]
    )
    assert np.allclose(values, exact, atol=TOLERANCE)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
def test_a_block_too_wide_to_route_falls_back_to_narrow_blocks(backend):
    """Falling back is not the same as giving up on communicating.

    Two wires that cannot exchange their outcomes as one block still can as two blocks
    of one, at gamma 3 each against the 4 of the local decomposition. Planning has to
    retry the group in smaller pieces rather than drop straight to no bundling.

    IQM's transpiler is the path that needs it: it refuses a custom gate, so the block
    cannot be handed to it as one marker as wide as itself, and the wires it lands on
    are whatever the layout chose. The qiskit path keeps the whole block and is covered
    by :func:`test_a_wide_block_can_survive_subcircuit_transpilation`.
    """
    marked, _plain, observables = _locc_pair()

    def plan(communication):
        cut_circuit = ck.get_locations_and_subcircuits(
            marked.copy(), options=CutOptions(wire_cut_communication=communication)
        )
        transpiled = ck.transpile_subcircuits(
            cut_circuit, backend, optimization_level=3, use_iqm_transpiler=True
        )
        return transpiled, ck.get_experiment_circuits(transpiled, observables)

    transpiled, forced = plan("always")
    _, local = plan("never")

    bundles = plan_bundles(
        transpiled.cut_locations,
        transpiled.subcircuits,
        transpiled.options,
        fits=coupling_filter(
            transpiled.cut_locations, transpiled.subcircuits, transpiled.backend
        ),
    )
    assert [bundle.kind for bundle in bundles] == ["cc_wire", "cc_wire"]
    assert all(bundle.size == 1 for bundle in bundles)
    assert forced.num_circuits < local.num_circuits
    _accepted_by(backend, forced)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
@pytest.mark.parametrize("level", [0, 3])
def test_a_wide_block_can_survive_subcircuit_transpilation(backend, level):
    """A block is routed properly when the transpiler is told its real width.

    Its placeholders are one qubit each, so left alone the transpiler lays them out as
    unrelated single-qubit gates and the block goes in afterwards on wires the device
    may not couple. Replacing them by one marker as wide as the block makes routing
    place it on a locus that exists, and nothing can be scheduled inside a single
    instruction, so the cuts it covers stay simultaneous.

    Falling back is always allowed -- the veto in ``coupling_filter`` sees to that -- so
    this asserts only that the answers are right and the device takes them, and that at
    least one configuration does keep the block.
    """
    marked, plain, observables = _locc_pair()

    def build(communication):
        cut_circuit = ck.get_locations_and_subcircuits(
            marked.copy(), options=CutOptions(wire_cut_communication=communication)
        )
        transpiled = ck.transpile_subcircuits(
            cut_circuit, backend, optimization_level=level, use_iqm_transpiler=False
        )
        return ck.get_experiment_circuits(transpiled, observables)

    experiment = build("auto")
    _accepted_by(backend, experiment)

    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    values = np.array(ck.estimate_expectation_values(results))
    exact = np.array(
        [
            float(np.real(Statevector(plain).expectation_value(pauli)))
            for pauli in observables.paulis
        ]
    )
    assert np.allclose(values, exact, atol=TOLERANCE)
    assert experiment.num_circuits <= build("never").num_circuits
