from __future__ import annotations

from collections import namedtuple
from copy import deepcopy

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator

from QCut.backend_utility import transpile_experiments
from QCut.cutcircuit import CutCircuit
from QCut.cutlocation import SingleQubitCutLocation
from QCut.qcuterror import QCutError
from QCut.wirecut import (
    estimate_expectation_values,
    get_experiment_circuits,
    run_experiments,
)

BitLocations = namedtuple("BitLocations", ("index", "registers"))


def _get_cut_locations(circuit):
    index = 0  # index of the current instruction in circuit_data
    circuit_data = circuit.data
    cut_locations = np.array([])

    # loop through circuit instructions
    # if operation is a Cut() instruction remove it and add registers and
    # offset index to cut_locations

    # rename varibales to be more descriptive (namely qs)
    while index < len(circuit):
        if circuit_data[index].operation.name == "Cut":
            # find qubits for Cut operation
            qubits = [
                circuit.find_bit(qubit).registers[0]
                for qubit in circuit_data[index].qubits
            ]

            # remove the cut operation
            circuit_data.remove(circuit_data[index])

            # append to cut_locations
            cut_locations = np.append(
                cut_locations, SingleQubitCutLocation((qubits[0], index))
            )

            # adjust index to account for removed operation
            index -= 1
        index += 1

    return cut_locations


def _insert_cut_nodes(circuit, cut_locations):
    placeholder_locations = []
    circuit_data = circuit.data
    cut_index = 0
    offset = 0
    for cut_location in cut_locations:
        cur_placeholder = ()
        measure_node = QuantumCircuit(1, name=f"Meas_{cut_index}").to_instruction()
        initialize_node = QuantumCircuit(1, name=f"Init_{cut_index}").to_instruction()
        cut_index += 1

        circuit_data.insert(
            cut_location.index + offset,
            CircuitInstruction(
                operation=measure_node,
                qubits=[Qubit(cut_location.qubits[0], cut_location.qubits[1])],
            ),
        )
        meas_plcaholder = cut_location.index + offset
        circuit_data.insert(
            cut_location.index + offset + 1,
            CircuitInstruction(
                operation=initialize_node,
                qubits=[Qubit(cut_location.qubits[0], cut_location.qubits[1])],
            ),
        )
        init_placeholder = cut_location.index + offset + 1

        cur_placeholder = (meas_plcaholder, init_placeholder)
        placeholder_locations.append(cur_placeholder)

        offset += 2

    return circuit, placeholder_locations


def _move_to_new_wire(circuit, num_cuts):  # noqa: C901
    count = 0
    # qr = QuantumRegister(num_cuts, "qpd")
    # circuit.qregs.append(qr)
    i = 0
    j = 0
    all_cut_qubits = []
    while i < len(circuit):
        ind, op = list(enumerate(circuit.data))[i]
        if "Meas" in op.operation.name:
            consecutive_cuts = 0
            cut_qubits = []
            j = 0
            while "Meas" in circuit.data[ind + j].operation.name:
                cut_qubits.append(circuit.data[ind + j].qubits[0])
                all_cut_qubits.append(
                    circuit.find_bit(circuit.data[ind + j].qubits[0]).index
                )
                consecutive_cuts += 1
                j += 2
            for _ in range(consecutive_cuts):
                q = Qubit()
                circuit.add_bits([q])
                circuit.qubits.remove(q)
                circuit.qubits.insert(circuit.qubits.index(cut_qubits[-1]) + 1, q)
            check = 0
            for subind, subop in enumerate(circuit.data[ind::]):
                if "Meas" in subop.operation.name and check < consecutive_cuts:
                    check += 1
                    continue
                new_qubs = []
                for qub in subop.qubits:
                    if qub in cut_qubits:
                        q_ind = cut_qubits.index(qub) + 1
                        new_qubs.append(
                            circuit.qubits[circuit.qubits.index(cut_qubits[-1]) + q_ind]
                        )
                    else:
                        new_qubs.append(qub)
                if new_qubs != subop.qubits:
                    circuit.data[ind + subind] = CircuitInstruction(
                        subop.operation, new_qubs
                    )

            count += consecutive_cuts
        if j != 0:
            i += j
            j = 0
        else:
            i += 1
    return circuit


def count_gates(qc: QuantumCircuit):
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def _remove_idle_wires(qc: QuantumCircuit):
    qc_out = deepcopy(qc)
    gate_count = count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
            for i in qc_out.qregs:
                if qubit in i._bits:
                    i._bits.remove(qubit)
    qc_out.qregs[0]._bit_indices = {
        qubit: qc_out.qubits.index(qubit) for qubit in qc_out.qubits
    }
    qc_out.qregs[0]._bits = qc_out.qubits
    qc_out.qregs[0]._size = len(qc_out.qregs[0]._bits)
    return qc_out


def _separate_subcircuits(circuit):
    dag = circuit_to_dag(circuit)

    circs = dag.separable_circuits()

    new_circs = []
    for i in circs:
        circ = _remove_idle_wires(dag_to_circuit(i))
        if len(circ.qubits) == 0:
            continue
        new_circs.append(circ)

    return new_circs


def _add_cbits(subcircuits):
    for circ in subcircuits:
        clbits = 0
        for i in circ:
            if "Meas" in i.operation.name:
                clbits += 1
        circ.add_register(ClassicalRegister(clbits, "qpd_meas"))
        circ.add_register(ClassicalRegister(circ.num_qubits - clbits, "meas"))

    return subcircuits


def get_qubit_map(subcircuits: list[QuantumCircuit]):
    def filter_obs_i(qc_data):
        return [i for i in qc_data if "obs" in i.operation.name]

    def sort_func(obs):
        return int(obs.operation.name.split("_")[1])

    map_qubit = {}
    count = 0
    for ind, i in enumerate(reversed(subcircuits)):
        for j in sorted(filter_obs_i(i.data), key=sort_func, reverse=True):
            map_qubit[int(j.operation.name.split("_")[1])] = count
            count += 1

    return map_qubit


def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
):
    """Get cut locations and subcircuits with placeholder operations.

    Args:
        circuit (QuantumCircuit): circuit with cuts inserted

    Returns:
        tuple: A tuple containing:
            - list[SingleQubitCutLocation]: Locations of the cuts as a list
            - list[QuantumCircuit]: Subcircuits with placeholder operations

    """
    circuit = circuit.copy()  # copy to avoid modifying the original circuit
    for i in range(circuit.num_qubits):
        obs_m = QuantumCircuit(1, name=f"obs_{i}")
        obs_m = obs_m.to_instruction()
        circuit.append(obs_m, [i])
    cut_locations = _get_cut_locations(circuit)
    circuit1, _placeholder_locations = _insert_cut_nodes(circuit, cut_locations)
    circuit = _move_to_new_wire(circuit1.copy(), len(cut_locations))
    subcircuits = _separate_subcircuits(circuit)
    subcircuits = _add_cbits(subcircuits)
    fixed_circs = []
    for i in subcircuits:
        test = QuantumCircuit(i.num_qubits)
        test.add_register(i.cregs[0])
        test.add_register(i.cregs[1])

        for j in i.data:
            qubits = [test.qubits[i.qubits.index(q)] for q in j.qubits]
            test.append(CircuitInstruction(j.operation, qubits))

        fixed_circs.append(test)
    if len(fixed_circs) <= 1:
        raise QCutError(
            "Invalid cuts. Check documentation to see how cuts should be placed."
        )

    map_qubits = get_qubit_map(fixed_circs)

    return cut_locations, fixed_circs, map_qubits


def run_cut_circuit(
    subcircuits: list[QuantumCircuit],
    cut_locations: np.ndarray[SingleQubitCutLocation],
    observables: list[int | list[int]],
    map_qubits: dict[int, int],
    backend=AerSimulator(),
    mitigate: bool = False,
) -> np.ndarray[float]:
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
        subcircuits (list[QuantumCircuit]):
            subcircuits containing the placeholder operations
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z observable)
        backend: backend to use for running experiment circuits (optional)
        mitigate (bool): wether or not to use readout error mitigation (optional)

    Returns:
        list: a list of expectation values

    """
    subexperiments, coefs, id_meas = get_experiment_circuits(subcircuits, cut_locations)
    if not isinstance(backend, AerSimulator):
        subexperiments = transpile_experiments(subexperiments.circuits, backend)
        subexperiments = CutCircuit(subexperiments)
    results = run_experiments(
        subexperiments,
        cut_locations,
        id_meas=id_meas,
        backend=backend,
        mitigate=mitigate,
    )

    return estimate_expectation_values(
        results, coefs, cut_locations, observables, map_qubits
    )


def run(
    circuit: QuantumCircuit,
    observables: list[int, list[int]],
    backend=AerSimulator(),
    mitigate: bool = False,
) -> list[float]:
    """Run the whole circuit knitting sequence with one function call.

    Args:
        circuit (QuantumCircuit): circuit with cut experiments
        observables (list[int | list[int]]):
            list of observbles in the form of qubit indices (Z-obsevable).
        backend: backend to use for running experiment circuits (optional)
        mitigate (bool): wether or not to use readout error mitigation (optional)

    Returns:
        list: a list of expectation values

    """
    # circuit = circuit.copy()
    qss, circs, map_qubits = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(circs, qss, observables, map_qubits, backend, mitigate)
