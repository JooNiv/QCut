from __future__ import annotations

from collections import namedtuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import CircuitError, CircuitInstruction, Clbit, Qubit
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


def num_parallel_cuts_placeholders(data, ind):
    num_placeholders = 0
    for i in data[ind:]:
        if "Meas_" in i[0].name or "Init_" in i[0].name:
            num_placeholders += 1
        else:
            break
    return num_placeholders


def get_subcircuit_qubits(circuit, subcircuits):
    qubits = []
    sub_arr = set()
    for i in subcircuits:
        for j in i:
            for q in j.qubits:
                sub_arr.add(circuit.find_bit(q).index)
        qubits.append(sub_arr)
        sub_arr = set()
    return qubits


def merge_subcircuits(circuit, subcircuits, cut_locations):
    cut_qubits = {cut.meas for cut in cut_locations}
    qubits_per_sub = get_subcircuit_qubits(circuit, subcircuits)
    merged = []
    merged_indices = set()
    for ind1, i in enumerate(qubits_per_sub):
        for ind2, j in enumerate(qubits_per_sub):
            if (
                ind1 != ind2
                and (ind1, ind2) not in merged_indices
                and (ind2, ind1) not in merged_indices
                and not i == cut_qubits
                and not j == cut_qubits
            ):
                if i == j or i.issubset(j) or j.issubset(i):
                    merged.append(subcircuits[ind1] + subcircuits[ind2])
                    merged_indices.add((ind1, ind2))
    result = []
    merged_set = set(merged_indices)
    merged_dict = {min(pair): merged[ind] for ind, pair in enumerate(merged_indices)}
    for ind, subcircuit in enumerate(subcircuits):
        if not any(ind in pair for pair in merged_set):
            result.append((subcircuit, False))
        elif ind in merged_dict:
            result.append((merged_dict[ind], True))
    return result


def _get_subcircuit_data(circuit, placeholder_locations):
    subcircuits = []
    cur_circ = []
    num_to_skip = 0
    prev_ind = 0
    for placeholder_location in placeholder_locations:
        if num_to_skip > 0:
            num_to_skip -= 1
            continue
        num_to_skip = max(
            0,
            num_parallel_cuts_placeholders(circuit.data, placeholder_location[0]) / 2
            - 1,
        )

        cur_circ = (
            cur_circ + circuit.data[prev_ind : placeholder_location[0] + 1]
            if num_to_skip == 0
            else cur_circ
            + circuit.data[prev_ind : placeholder_location[0] + 1]
            + [circuit.data[placeholder_location[0] + int(num_to_skip * 2)]]
        )
        subcircuits.append(cur_circ)
        cur_circ = []
        for i in range(0, int(num_to_skip * 2), 2):
            cur_circ.append(circuit.data[placeholder_location[0] + 1 + i])
        prev_ind = int(placeholder_location[0] + num_to_skip * 2 + 1)
    cur_circ = cur_circ + circuit.data[prev_ind:]
    subcircuits.append(cur_circ)
    return subcircuits


def expand_subcircuit(circuit):
    cut_indices = cut_wire_indices(circuit)
    new_sub0 = QuantumCircuit(circuit.num_qubits + num_cuts_plcaholder(circuit))
    offset = 0
    cut_index = 0
    for i in circuit.data:
        if "Meas_" in i[0].name:
            measure_node = QuantumCircuit(1, name=f"Meas_{cut_index}").to_instruction()
            qubits_for_operation = [
                Qubit(circuit.qregs[0], circuit.find_bit(x).index) for x in i.qubits
            ]
            new_sub0.append(
                measure_node,
                [
                    Qubit(new_sub0.qregs[0], circuit.find_bit(x).index)
                    for x in qubits_for_operation
                ],
            )
            offset += 1
            cut_index += 1
        elif "Init_" in i[0].name:
            initialize_node = QuantumCircuit(
                1, name=f"Init_{cut_index}"
            ).to_instruction()
            qubits_for_operation = [
                Qubit(circuit.qregs[0], circuit.find_bit(x).index) for x in i.qubits
            ]
            new_sub0.append(
                initialize_node,
                [
                    Qubit(new_sub0.qregs[0], circuit.find_bit(x).index + offset)
                    for x in qubits_for_operation
                ],
            )
            cut_index += 1
        else:
            qubits_for_operation = [
                Qubit(circuit.qregs[0], circuit.find_bit(x).index) for x in i.qubits
            ]
            new_sub0.data.append(
                CircuitInstruction(
                    i.operation,
                    [
                        Qubit(new_sub0.qregs[0], circuit.find_bit(x).index + offset)
                        if circuit.find_bit(x).index in cut_indices
                        else Qubit(new_sub0.qregs[0], circuit.find_bit(x).index)
                        for x in qubits_for_operation
                    ],
                )
            )

    return new_sub0


def num_cuts_plcaholder(circuit):
    num_cuts = 0
    for i in circuit.data:
        if "Meas_" in i[0].name:
            num_cuts += 1
    return num_cuts


def cut_wire_indices(circuit):
    cut_indices = []
    for ind, i in enumerate(circuit.data):
        if "Meas_" in i[0].name:
            cut_indices.append(circuit.find_bit(i.qubits[0]).index)
    return cut_indices


def build_subcircuits(circuit, merged_subcircuits):
    circuits = []
    merged = [i[0] for i in merged_subcircuits]
    qubits = get_subcircuit_qubits(circuit, merged)
    num_qubits_per_circuit = [len(i) for i in qubits]
    clbits = 0
    for ind, pair in enumerate(merged_subcircuits):
        i = pair[0]
        check = pair[1]
        offset = min(qubits[ind])
        subcirc = QuantumCircuit(num_qubits_per_circuit[ind])
        for j in i:
            if "Meas_" in j.operation.name:
                clbits += 1
            qubits_for_operation = [
                Qubit(circuit.qregs[0], circuit.find_bit(x).index) for x in j.qubits
            ]
            subcirc.data.append(
                CircuitInstruction(
                    j.operation,
                    [
                        Qubit(subcirc.qregs[0], circuit.find_bit(x).index - offset)
                        for x in qubits_for_operation
                    ],
                )
            )
        if check:
            subcirc = expand_subcircuit(subcirc)
        cr_qpd = ClassicalRegister(clbits, name="qpd_meas")
        cr = ClassicalRegister(subcirc.num_qubits - clbits, name="meas")
        subcirc.cregs.append(cr_qpd)
        subcirc.cregs.append(cr)
        clbit_indices = {}
        for cl in range(clbits):
            bit = Clbit(cr_qpd, cl)
            subcirc.clbits.append(bit)
            clbit_indices[bit] = BitLocations(cl, [(cr_qpd, cl)])
        for cl in range(subcirc.num_qubits - clbits):
            bit = Clbit(cr, cl)
            subcirc.clbits.append(bit)
            clbit_indices[bit] = BitLocations(cl, [(cr, cl)])
        subcirc._clbit_indices = clbit_indices
        clbits = 0
        circuits.append(subcirc)

    return circuits


def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
) -> tuple[list[SingleQubitCutLocation], list[QuantumCircuit]]:
    """Get cut locations and subcircuits with placeholder operations.

    Args:
        circuit (QuantumCircuit): circuit with cuts inserted

    Returns:
        tuple: A tuple containing:
            - list[SingleQubitCutLocation]: Locations of the cuts as a list
            - list[QuantumCircuit]: Subcircuits with placeholder operations

    """
    circuit = circuit.copy()  # copy to avoid modifying the original circuit
    cut_locations = _get_cut_locations(circuit)
    circ, placeholder_locations = _insert_cut_nodes(circuit, cut_locations)
    subcircuits = _get_subcircuit_data(circ, placeholder_locations)
    merged = merge_subcircuits(circ, subcircuits)
    try:
        subcircuits = build_subcircuits(circ, merged)
    except CircuitError as e:
        msg = "Invalid cut placement. See documentation for how cuts should be placed."
        raise QCutError(msg) from e

    return cut_locations, subcircuits


def run_cut_circuit(
    subcircuits: list[QuantumCircuit],
    cut_locations: np.ndarray[SingleQubitCutLocation],
    observables: list[int | list[int]],
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

    return estimate_expectation_values(results, coefs, cut_locations, observables)


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
    circuit = circuit.copy()
    qss, circs = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(circs, qss, observables, backend, mitigate)
