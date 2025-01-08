"""Circuit knitting wire cut functionality.

.. deprecated:: 0.2.0
   This module is deprecated and will be removed in version 0.3.0.
   Use `single_qubit_wirecut` instead.
"""

from __future__ import annotations

import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitError, CircuitInstruction, Qubit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator

from QCut.backend_utility import transpile_experiments
from QCut.cutcircuit import CutCircuit
from QCut.cutlocation import CutLocation
from QCut.qcuterror import QCutError
from QCut.wirecut import (
    estimate_expectation_values,
    get_experiment_circuits,
    run_experiments,
)

warnings.simplefilter("always", DeprecationWarning)
warnings.warn(
    """
    The module `two_qubit_wirecut` for wirecuts as two-qubit
    gates is deprecated as of 0.2.0 and will be removed in 0.3.0.
    Please use `single_qubit_wirecut` instead. 'single_qubit_wirecut' will be
    renamed to 'wirecut' in a future version. For migration see documentation.
    """,
    DeprecationWarning,
    stacklevel=2,
)

ERROR = 0.0000001

def _get_cut_locations(circuit: QuantumCircuit) -> np.ndarray[CutLocation]:
    """Get the locations of the cuts in the circuit.

    Iterate circuit data. When find cut instruction, save indices of
    qubits for operation. 0-qubit measure channel, 1-qubit initialize channel.
    Also save index of the cut instruction in the circuit.

    CutLocation class has form [meas qubit: _, init qubit: _, cut index: _].

    Note:
    ----
        get_cut_location modifies the circuit passed as an argument by removing cut_wire
        gates. Therefore trying to run it multiple times in a row on the same circuit
        will fail. To avoid this it is not recommended to run this function by itself.
        Instead use get_locations_and_subcircuits().

    Args:
        circuit (QuantumCircuit): Quantum circuit with cut_wire operations.

    Returns:
        np.ndarray[CutLocation]:
            Locations of the cuts as a numpy array.

    Raises:
        QCutError: if no cuts found

    """
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
                cut_locations, CutLocation((tuple(qubits), index))
            )

            # adjust index to account for removed operation
            index -= 1
        index += 1

    # if no cuts found raise error
    if len(cut_locations) == 0:
        exc = """No cuts in circuit. Did you pass the wrong circuit or try to run
                get_cut_location() multiple times in a row?"""
        raise QCutError(exc)
    return cut_locations


def _get_bounds(cut_locations: list[CutLocation]) -> list:
    """Get the bounds for subcircuits as qubit indices.

    Args:
        cut_locations (list[CutLocation]): Locations of the cuts as a list.

    Returns:
        list:
            Bounds as a list of qubit indices.

    """

    def _add_cut_to_group(cut: CutLocation, group: list) -> None:
        """Update the group with the new cut information."""
        group[0][0] = max(group[0][0], cut.meas)
        group[0][1] = max(group[0][1], cut.init)
        group[2].append([cut.meas, cut.init])

    def _extend_or_create_group(cut: CutLocation, cut_groups: list) -> None:
        """Add the cut to an existing group or create a new group."""
        
        for group in cut_groups:
            if cut.index in group[1]:
                _add_cut_to_group(cut, group)
                return
            elif min(cut.meas, cut.init) - max(group[0]) < 0:
                group[0][0] = (
                    min(group[0][0], cut.meas, cut.init)
                    if group[0][0] == min(group[0])
                    else max(group[0][0], cut.meas, cut.init)
                )

                group[0][1] = (
                    min(group[0][1], cut.meas, cut.init)
                    if group[0][1] == min(group[0])
                    else max(group[0][1], cut.meas, cut.init)
                )
                group[1].append(cut.index)
                group[2].append([cut.meas, cut.init])
                return

        cut_groups.append(
            ([cut.meas, cut.init], [cut.index], [[cut.meas, cut.init]])
        )

    cut_groups = []

    for index, cut in enumerate(cut_locations):
        if index == 0:
            cut_groups.append(
                ([cut.meas, cut.init], [cut.index], [[cut.meas, cut.init]])
            )
        else:
            _extend_or_create_group(cut, cut_groups)
    bounds = [max(min(x) for x in group[2]) for group in cut_groups]

    return bounds


def get_locations_and_bounds(
    circuit: QuantumCircuit,
) -> tuple[np.ndarray[CutLocation], list[int]]:
    """Get the locations of the cuts in the circuit and the subcircuit bounds.

    Args:
        circuit (QuantumCircuit): Quantum circuit with Move() operations.

    Returns:
        tuple: A tuple containing:
            - np.ndarray[CutLocation]: Locations of the cuts as a numpy array.
            - list[int]: Bounds as a list.
    """
    cut_locations = _get_cut_locations(circuit)
    bounds = _get_bounds(cut_locations)

    return cut_locations, bounds


def _insert_meassure_prepare_channel(
    circuit: QuantumCircuit, cut_locations: np.ndarray[CutLocation]
) -> QuantumCircuit:
    """Insert the measure and initialize nodes at the cut locations.

    Loop through circuit. When cut found remove it and insert placeholder measure
    channel in place of 0-qubit of cut and initialize channel in place of 1-qubit.
    Placeholder channels are named "Meas_{ind}" and "Init_{ind}". ind is the index of
    cut in circuit. As in first cut has in 0, second 1 and so on. To ensure correct
    order of operations if meas qubit index is larger than init qubit index of the cut
    the measure channel placeholder gets inserted first, and the other way around.

    Args:
        circuit (QuantumCircuit): Quantum circuit with Move() operations.
        cut_locations (np.ndarray[CutLocations]): Locations of the cuts as a list.

    Returns:
        QuantumCircuit:
            circuit with measure and initialize nodes inserted

    Raises:
        QCutError: If the cuts are not valid.

    """

    circuit_data = circuit.data
    offset = 0
    for index, cut in enumerate(cut_locations):
        # Placeholder measure node operation
        measure_node = QuantumCircuit(1, name=f"Meas_{index}").to_instruction()

        # Placeholder initialize node operation
        initialize_node = QuantumCircuit(1, name=f"Init_{index}").to_instruction()

        # Determine which operation to insert first based on cut location
        if max(cut.meas, cut.init) == cut.meas:
            circuit_data.insert(
                cut.index + offset,
                CircuitInstruction(
                    operation=measure_node,
                    qubits=[Qubit(cut.qubits[0][0], cut.qubits[0][1])],
                ),
            )

            circuit_data.insert(
                cut.index + offset,
                CircuitInstruction(
                    operation=initialize_node,
                    qubits=[Qubit(cut.qubits[1][0], cut.qubits[1][1])],
                ),
            )
        else:
            circuit_data.insert(
                cut.index + offset,
                CircuitInstruction(
                    operation=initialize_node,
                    qubits=[Qubit(cut.qubits[1][0], cut.qubits[1][1])],
                ),
            )

            circuit_data.insert(
                cut.index + offset,
                CircuitInstruction(
                    operation=measure_node,
                    qubits=[Qubit(cut.qubits[0][0], cut.qubits[0][1])],
                ),
            )

        # Update the offset since we have inserted two instructions
        offset += 2

    return circuit


def _build_subcircuit(
    current_bound: int,
    previous_bound: int,
    clbits: int,
    subcircuit_operations: list[CircuitInstruction],
    circuit: QuantumCircuit,
    last: bool = False,
) -> QuantumCircuit:
    """Help build subcircuits.

    Args:
        circuit (QuantumCircuit): Quantum circuit with Move() operations.
        previous_bound (int): last qubit of previous subcircuit.
        current_bound (int): last qubit of current subcircuit.
        clbits (int): number of classical bits.
        subcircuit_operations (list[CircuitInstruction]):
            operations for subcircuit to be built.
        last (bool): change behaviour if last subcircuit

    Returns:
        QuantumCircuit:
            The built subcircuit.

    """
    qr_size = (
        current_bound - previous_bound if last else current_bound - previous_bound + 1
    )
    # define quantum register
    qr = QuantumRegister(qr_size)

    # define classical registers for obsrvable - and qpd - measurements
    crqpd = ClassicalRegister(clbits, "qpd_meas")
    cr = ClassicalRegister(qr_size - clbits, "meas")

    subcircuit = QuantumCircuit(qr, crqpd, cr)  # initialize the subcircuit

    for operation in subcircuit_operations:  # loop throgh the subcircuit_operations and
        # add them to the subcircuit
        # get the qubits needed fot the operation and bind them to the quantum register
        # of the subcircuit
        if last:
            qubits_for_operation = [
                Qubit(qr, circuit.find_bit(qubit).index - current_bound)
                for qubit in operation.qubits
            ]
        else:
            qubits_for_operation = [
                Qubit(qr, circuit.find_bit(qubit).index - previous_bound)
                for qubit in operation.qubits
            ]

        # insert operation to subcricuit
        subcircuit.append(operation.operation, qubits_for_operation)

    return subcircuit


# Cuts the given circuit into two at the location of the cut marker
def _separate_sub_circuits(
    circuit: QuantumCircuit, sub_circuit_qubit_bounds: list[int]
) -> list[QuantumCircuit]:
    """Split the circuit with placeholder measure and prepare channels into separate
    subcircuits.

    Iterate over circuit data of circuit with placeholder operations. Insert num_qubits
    to bounds. Set current bound to first element in vounds. Insert all operations to a
    new circuit. Once hit operation with qubit that has index >= current bound, store
    subciruit to an array, initialize a new circuit and repeat untill all operartions
    have been iterated over. If meas inserted one has to add a cbit to a classical
    register qpd_meas to store the qpd basis measurements.

    Args:
        circuit (QuantumCircuit): Quantum circuit with cut_wire() operations.
        sub_circuit_qubit_bounds (list[int]):
            Bounds for subcircuits as list of qubit indices.

    Returns:
    -------
        list[QuantumCiruit]:
            Array of subcircuits

    """
    # remove barriers
    circuit = RemoveBarriers()(circuit)

    # append final bound
    sub_circuit_qubit_bounds.append(circuit.num_qubits)

    subcircuits_list = [0] * len(sub_circuit_qubit_bounds)  # initialize solution array
    current_subcircuit = 0  # counter for which subcircuit we are in
    clbits = 0  # number of classical bits needed for subcircuit
    previous_bound = 0  # previous bound
    subcircuit_operations = []  # array for collecting subcircuit operations

    for i, op in enumerate(circuit.data):
        qubits = [circuit.find_bit(x).index for x in op.qubits]  # qubits in operation

        if "Meas" in op.operation.name:
            clbits += 1  # if measure node add a classical bit

        if i == len(circuit.data) - 1:  # if at the end of the original circuit, handle
            # final subcircuit

            subcircuit_operations.append(op)  # append the final operation to list
            subcircuit = _build_subcircuit(
                sub_circuit_qubit_bounds[current_subcircuit],
                previous_bound,
                clbits,
                subcircuit_operations,
                circuit,
                last=True,
            )

            subcircuits_list[current_subcircuit] = subcircuit
            return subcircuits_list

        # if sub_circuit_qubit_bounds[current_subcircuit] in qubits:
        if any(
            qubit > sub_circuit_qubit_bounds[current_subcircuit] for qubit in qubits
        ):
            # build the subcircuit
            subcircuit = _build_subcircuit(
                sub_circuit_qubit_bounds[current_subcircuit],
                previous_bound,
                clbits,
                subcircuit_operations,
                circuit,
            )
            subcircuits_list[current_subcircuit] = subcircuit

            # reset variables
            subcircuit_operations = []
            clbits = 0
            previous_bound = sub_circuit_qubit_bounds[current_subcircuit] + 1
            current_subcircuit += 1

        subcircuit_operations.append(op)
    return subcircuits_list

def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
) -> tuple[list[CutLocation], list[QuantumCircuit]]:
    """Get cut locations and subcircuits with placeholder operations.

    Args:
        circuit (QuantumCircuit): circuit with cuts inserted

    Returns:
        tuple: A tuple containing:
            - list[CutLocation]: Locations of the cuts as a list
            - list[QuantumCircuit]: Subcircuits with placeholder operations

    """
    circuit = circuit.copy()  # copy to avoid modifying the original circuit
    cut_locations = _get_cut_locations(circuit)
    sorted_cut_locations = sorted(cut_locations, key=lambda x: min(x.meas, x.init))
    circ = _insert_meassure_prepare_channel(circuit, cut_locations)
    bounds = _get_bounds(sorted_cut_locations)
    try:
        subcircuits = _separate_sub_circuits(circ, bounds)
    except CircuitError as e:
        msg = "Invalid cut placement. See documentation for how cuts should be placed."
        raise QCutError(msg) from e

    return sorted_cut_locations, subcircuits

def run_cut_circuit(
    subcircuits: list[QuantumCircuit],
    cut_locations: np.ndarray[CutLocation],
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
