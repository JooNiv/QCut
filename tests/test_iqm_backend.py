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


CASES = {"gate_cut": _gate_cut, "wire_cut": _wire_cut, "locc_block": _locc_block}


def _every_register_is_written(circuit):
    """IQMBackend.run refuses a job whose circuit has a register nothing writes to."""
    utilisation = dict.fromkeys(circuit.cregs, 0)
    for instruction in circuit.data:
        if instruction.clbits:
            register = circuit.find_bit(instruction.clbits[0]).registers[0][0]
            utilisation[register] += 1
    return 0 not in utilisation.values()


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
