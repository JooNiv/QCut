"""Circuit knitting wire cut functionality."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitError, CircuitInstruction, Qubit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator
from qiskit_experiments.library import LocalReadoutError

from QCut.backend_utility import transpile_experiments
from QCut.identity_qpd import identity_qpd

if TYPE_CHECKING:
    from collections.abc import Iterable

ERROR = 0.0000001


class QCutError(Exception):
    """Exception raised for custom error conditions.

    Attributes
    ----------
        message (str): Explanation of the error.
        code (int, optional): Error code representing the error type.

    """

    def __init__(
        self, message: str = "An error occurred", code: int | None = None
    ) -> None:
        """Init.

        Args:
        -----
            message: Explanation of the error. Default is "An error occurred".
            code: Optional error code representing the error type.

        Attributes:
        -----------
            message (str): The error message provided during initialization.
            code (int or None): The error code provided, or None if not specified.

        """
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the string representation of the error.

        Returns
        -------
            str: A string describing the error, including the code if available.

        """
        if self.code:
            return f"[Error {self.code}] {self.message}"

        return self.message


# Class for storing results from single sub-circuit run


class SubResult:
    """Storage class for easier storage/access to the results of a subcircuit."""

    def __init__(self, measurements: list, count: int) -> None:
        """Init."""
        self.measurements = measurements  # measurement results
        self.count = count  # counts for this specific measurement

    def __str__(self) -> str:
        """Format string."""
        return f"{self.measurements}, {self.count}"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


# Store total results of all sub-circuits (two for now)
class TotalResult:
    """Storage class for easier access to the results of a subcircuit group."""

    def __init__(self, *subcircuits: list[SubResult]) -> None:
        """Init."""
        self.subcircuits = subcircuits

    def __str__(self) -> str:
        """Format string."""
        substr = ""
        for i in self.subcircuits:
            substr += f"{i}"
        return substr

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


class CutLocation:
    """Storage class for storing cut locations."""

    def __init__(
        self, cut_location: tuple[tuple[tuple[QuantumRegister, int]], int]
    ) -> None:
        """Init."""
        self.qubits = cut_location[0]
        self.meas = cut_location[0][0][1]
        self.init = cut_location[0][1][1]
        self.index = cut_location[1]

    def __eq__(self, other: CutLocation) -> bool:
        """Equality."""
        if not isinstance(other, CutLocation):
            return NotImplemented

        return (
            self.meas == other.meas
            and self.init == other.init
            and self.index == other.index
        )

    def __str__(self) -> str:
        """Format string."""
        msg = (
            f"meas qubit: {self.meas}, init qubit: {self.init}, "
            f"cut index: {self.index}"
        )
        return msg

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


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
    ----
        circuit: Quantum circuit with cut_wire operations.

    Returns:
    -------
        Locations of the cuts as a list.

    Raises:
    ------
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
    ----
        cut_locations: Locations of the cuts as a list.

    Returns:
    -------
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
                break
            if min(cut.meas, cut.init) - max(group[0]) < 0:
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
                break

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
    -----
        circuit: Quantum circuit with Move() operations.

    Returns:
    --------
        Locations of the cuts and bounds as a list.

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
    ----
        circuit: Quantum circuit with Move() operations.
        cut_locations: Locations of the cuts as a list.

    Returns:
    -------
        circuit with measure and initialize nodes inserted

    Raises:
    ------
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
    ----
        circuit: Quantum circuit with Move() operations.
        previous_bound: last qubit of previous subcircuit.
        current_bound: last qubit of current subcircuit.
        clbits: number of classical bits.
        subcircuit_operations: operations for subcircuit to be built.
        last: change behaviour if last subcircuit

    Returns:
    -------
        circs: Array of subcircuits
    Raises:

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
    ----
        circuit: Quantum circuit with Move() operations.
        sub_circuit_qubit_bounds: Bounds for subcircuits as list of qubit indices.

    Returns:
    -------
        circs: Array of subcircuits
    Raises:

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


def get_qpd_combinations(
    cut_locations: np.ndarray[CutLocation],
) -> Iterable[tuple[dict]]:
    """Get all possible combinations of the QPD operations so that each combination
    has len(cut_locations) elements.

    For a single cut operations can be straightforwardly inserted from the identity qpd.
    If multiple cuts are made one need to take the cartesian product of the identity
    qpd with itself n times, where n is number of cuts. This will give a qpd with
    8^n rows. Each row corresponds to a subcircuit group. These operations can then
    be inserted to generate the experiment circuits.

    Args:
    -----
        cut_locations: cut locations

    Returns:
    --------
        ops: list of the possible QPD operations

    Raises:

    """
    return product(identity_qpd, repeat=len(cut_locations))


def _adjust_cregs(subcircuit: QuantumCircuit) -> None:
    """Adjust classical registers for identity measurements."""
    if len(subcircuit.cregs) > 1:
        if subcircuit.cregs[0].size == 1:
            del subcircuit.clbits[subcircuit.cregs[0].size - 1]
            del subcircuit.cregs[0]._bits[subcircuit.cregs[0].size - 1]
            del subcircuit.cregs[0]
        else:
            del subcircuit.clbits[subcircuit.cregs[0].size - 1]
            del subcircuit.cregs[0]._bits[subcircuit.cregs[0].size - 1]
            subcircuit.cregs[0]._size -= 1


def _finalize_subcircuit(
    subcircuit: QuantumCircuit, qpd_qubits: list[int]
) -> QuantumCircuit:
    """Finalize the subcircuit by measuring remaining qubits and decomposing."""
    meas_qubits = [i for i in range(subcircuit.num_qubits) if i not in qpd_qubits]
    if len(subcircuit.cregs) >= 2:
        subcircuit.measure(meas_qubits, subcircuit.cregs[1])
    else:
        subcircuit.measure(meas_qubits, subcircuit.cregs[0])
    return subcircuit


def get_placeholder_locations(subcircuits: list[QuantumCircuit]) -> list:
    """Test."""
    ops = []
    for circ in subcircuits:
        subops = []
        for ind, op in enumerate(circ):
            if "Meas" in op.operation.name or "Init" in op.operation.name:
                subops.append((ind, op))
        ops.append(subops)

    return ops


def get_experiment_circuits(  # noqa: C901
    subcircuits: list[QuantumCircuit],  # noqa: C901
    cut_locations: np.ndarray[CutLocation],
) -> tuple[list[list[QuantumCircuit]], list[int], list[tuple[int, int, int]]]:
    """Generate experiment circuits by inserting QPD operations on
    measure/initialize nodes.

    Loop through qpd combinations. Calculate coefficient for subcircuit group by
    taking the product of all coefficients in the current qpd row. Loop through
    subcircuits generated in 4. Make deepcopy of subcircuit and iterate over its
    circuit data. When hit either Meas_{ind} of Init_{ind} repace it with operation
    found in qpd[ind]["op"/"init"]. WHile generating experiment circuits also
    generate a list of locations that have an identity basis measurement. These
    measurement outcomes need to be added during post-processing. Locations added as
    [index of experiment circuit, index of subcircuit, index of classical bit
    corresponding to measurement]. Repeat untill looped through all qpd rows.
    sircuits reutrned as [circuit_group0, circuit_group1, ...], where circuit_goup
    is [subciruit0, subcircuit1, ...].

    Args:
    -----
        subcircuits: subcircuits with measure/initialize nodes.
        cut_locations: cut locations.

    Returns:
    --------
        experimentCircuits: list of experiment circuits.
        coefficients: sign coefficients for each circuit.
        id_meas: list of index pointers to results that need additional post-processing
        due to identity basis measurement.

    """
    qpd_combinations = get_qpd_combinations(cut_locations)  # generate the QPD
    # operation combinations

    # initialize solution lists
    cuts = len(cut_locations)
    num_circs = np.power(8, cuts)
    experiment_circuits = []
    id_meas = np.full((num_circs, 3), None)
    num_id_meas = 0
    coefficients = np.empty(num_circs)
    placeholder_locations = get_placeholder_locations(subcircuits)
    for id_meas_experiment_index, qpd in enumerate(
        qpd_combinations
    ):  # loop through all
        # QPD combinations
        coefficients[id_meas_experiment_index] = np.prod([op["c"] for op in qpd])
        sub_experiment_circuits = []  # sub array for collecting related experiment
        # circuits
        inserted_operations = 0
        for id_meas_subcircuit_index, circ in enumerate(subcircuits):
            subcircuit = deepcopy(circ)
            offset = 0
            classical_bit_index = 0
            id_meas_bit = 0
            qpd_qubits = []  # store the qubit indices of qubits used for qpd 
                             # measurements
            for op_ind in placeholder_locations[id_meas_subcircuit_index]:
                ind, op = op_ind
                if "Meas" in op.operation.name:  # if measure channel remove placeholder
                    # and insert current
                    # qpd operation
                    qubit_index = subcircuit.find_bit(op.qubits[0]).index
                    subcircuit.data.pop(ind + offset)  # remove plaxceholder
                    # measure channel
                    qpd_qubits.append(qubit_index)  # store index
                    qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
                    meas_op = qpd[int(op.operation.name.split("_")[-1])]["op"]
                    if meas_op.name == "id-meas":  # if identity measure channel
                        # store indices
                        id_meas[num_id_meas] = np.array(
                            [
                                id_meas_experiment_index,
                                id_meas_subcircuit_index,
                                id_meas_bit,
                            ]
                        )
                        num_id_meas += 1
                        # remove extra classical bits and registers
                        _adjust_cregs(subcircuit)
                        for subop in reversed(meas_op.data):
                            subcircuit.data.insert(
                                ind + offset,
                                CircuitInstruction(
                                    operation=subop, qubits=qubits_for_operation
                                ),
                            )
                    else:
                        for i, subop in enumerate(reversed(meas_op.data)):
                            if i == 0:
                                subcircuit.data.insert(
                                    ind + offset,
                                    CircuitInstruction(
                                        operation=subop.operation,
                                        qubits=qubits_for_operation,
                                        clbits=[
                                            subcircuit.cregs[0][classical_bit_index]
                                        ],
                                    ),
                                )
                            else:
                                subcircuit.data.insert(
                                    ind + offset,
                                    CircuitInstruction(
                                        operation=subop.operation,
                                        qubits=qubits_for_operation,
                                    ),
                                )

                        # increment classical bit counter
                        classical_bit_index += 1

                    id_meas_bit += 1
                    inserted_operations += 1
                    offset += len(meas_op) - 1

                if "Init" in op.operation.name:
                    subcircuit.data.pop(ind + offset)
                    init_op = qpd[int(op.operation.name.split("_")[-1])]["init"]
                    qubits_for_operation = [
                        Qubit(subcircuit.qregs[0], subcircuit.find_bit(x).index)
                        for x in op.qubits
                    ]
                    for subop in reversed(init_op.data):
                        subcircuit.data.insert(
                            ind + offset,
                            CircuitInstruction(
                                operation=subop.operation, qubits=qubits_for_operation
                            ),
                        )

                    inserted_operations += 1
                    offset += len(init_op) - 1

            subcircuit = _finalize_subcircuit(subcircuit, qpd_qubits)
            sub_experiment_circuits.append(subcircuit)
        experiment_circuits.append(sub_experiment_circuits)

    return experiment_circuits, coefficients, id_meas[:num_id_meas]


def _run_mitigate(sub_result: list[tuple], shots: int, backend) -> list[tuple]:
    """Run experiment circuits and apply readout error mitigation."""
    numqubits_per_circ = set()
    for res in sub_result:
        measurements = list(res.keys())
        circ_numqubits = len(measurements[0].replace(" ", ""))
        numqubits_per_circ.add(circ_numqubits)

    mitigators = {
        circ_numqubits: LocalReadoutError(list(range(circ_numqubits)))
        .run(backend)
        .analysis_results("Local Readout Mitigator")
        .value
        for circ_numqubits in numqubits_per_circ
    }

    for ind, res in enumerate(sub_result):
        measurements = list(res.keys())
        circ_numqubits = list(range(len(measurements[0].replace(" ", ""))))
        mitigator = mitigators[len(circ_numqubits)]
        meas_bits = len(measurements[0].split(" ")[0])
        mitigated_quasi_probs = mitigator.quasi_probabilities(res)
        probs_test = {
            f"{int(old_key):0{len(circ_numqubits)}b}"[::-1][:meas_bits]
            + " "
            + f"{int(old_key):0{len(circ_numqubits)}b}"[::-1][
                meas_bits:
            ]: mitigated_quasi_probs[old_key] * shots
            if mitigated_quasi_probs[old_key] > 0
            else 0
            for old_key in mitigated_quasi_probs
        }

        sub_result[ind] = probs_test
    return sub_result


def run_experiments(
    experiment_circuits: list[list[QuantumCircuit]],
    cut_locations: np.ndarray[CutLocation],
    id_meas: list[tuple[int, int, int]],
    shots: int = 2**12,
    backend: None = None,
    mitigate: bool = False,
) -> list[TotalResult]:
    """Run experiment circuits.

    Loop through experiment circuits and then loop through circuit group and run each
    circuit. Store results as [group0, group1, ...] where group is [res0, res1, ...].
    where res is "xxx yy": count xxx are the measurements from the end of circuit
    measurements on the meas classical register and yy are the qpd basis measurement
    results from the qpd_meas class register.

    Args:
    -----
        experiment_circuits: experiment circuits
        cut_locations: list of cut locations
        id_meas: list of identity basis measurement locations
        shots: number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)
        mitigate: wether to use readout error mitigation or not (optional)

    Returns:
    --------
        processed_results: list of transformed results

    """
    cuts = len(cut_locations)
    # number of samples neede
    samples = int(np.power(4, (2) * cuts) / np.power(ERROR, 2))
    samples = int(samples / len(experiment_circuits))
    if backend is None:
        backend = AerSimulator()

    results = [0] * (len(experiment_circuits))

    for count, subcircuit_group in enumerate(experiment_circuits):
        sub_result = [
            backend.run(i, shots=shots).result().get_counts() for i in subcircuit_group
        ]
        if mitigate:
            sub_result = _run_mitigate(sub_result, shots, backend)

        results[count] = sub_result

        sub_result = []
    return _process_results(results, id_meas, shots, samples)


def _process_results(
    results: list[dict[str:int]],
    id_meas: list[tuple[int, int, int]],
    shots: int,
    samples: int,
) -> list:
    """Transform results with post processing function {0,1} -> [-1, 1].

    Tranform results so that we map 0 -> -1 and 1 -> 1. Give processed results in form
    [TotalResult0, TotalResult1, ...], where TotalResult is
    [SubResult0, SubResult1, ...] and SubResult are [[[x0,x0,x0], [y0,y0], counts0],
    [[x1,x1,x1], [y1,y1], counts1], ...].

    Args:
    ----
        results: results from experiment circuits
        id_meas: locations of identity basis measurements
        shots: number of shots per circuit run
        samples: number of needed samples

    Returns:
    -------
        processed_results: list of transformed results

    """
    preocessed_results = []
    for experiment_run in results:
        experiment_run_results = []
        for sub_result in experiment_run:
            circuit_results = []
            for meassurements, count in sub_result.items():
                # separate end measurements from mid-circuit measurements
                separate_measurements = meassurements.split(" ")

                # map to eigenvalues
                result_eigenvalues = [
                    np.array([-1 if x == "0" else 1 for x in i])
                    for i in separate_measurements
                ]
                circuit_results.append(
                    SubResult(result_eigenvalues, count / shots * samples)
                )
            experiment_run_results.append(circuit_results)
        preocessed_results.append(TotalResult(experiment_run_results))

    for loc in id_meas:
        for i in preocessed_results[loc[0]].subcircuits[0][loc[1]]:
            if len(i.measurements) == 1:
                i.measurements.append(np.array([-1]))
            else:
                i.measurements[1] = np.insert(i.measurements[1], loc[2], -1)
    return preocessed_results


# Calculate the approx expectation values for the original circuit
def estimate_expectation_values(
    results: list[TotalResult],
    coefficients: list[int],
    cut_locations: np.ndarray[CutLocation],
    observables: list[int | list[int]],
) -> list[float]:
    """Calculate the estimated expectation values.

    Loop through processed results. For each result group generate all products of
    different measurements from different subcircuits of the group. For each result
    from qpd measurements calculate qpd coefficient and from counts calculate weight.
    Get results for qubits corresponding to the observables. If multiqubit observable
    multiply individual qubit eigenvalues and multiply by (-1)^(m+1) where m is number
    of qubits in the observable. Multiply by weight and add to sub expectation value.
    Once all results iterated over move to next circuit group. Lastly multiply
    by 4^(2*n), where n is the number of cuts, and divide by number of samples.

    Args:
    -----
        results: results from experiment circuits
        coefficients: list of coefficients for each subcircuit group
        cut_locations: cut locations
        observables: observables to calculate expectation values for

    Returns:
    --------
        list: expectations as a list of floats list of floats

    """
    cuts = len(cut_locations)
    # number of samples neede
    samples = int(np.power(4, 2 * cuts) / np.power(ERROR, 2))
    shots = int(samples / len(results))

    # ininialize approx expectation values of an array of ones
    expectation_values = np.ones(len(observables))
    for experiment_run, coefficient in zip(results, coefficients):
        # add sub results to the total approx expectation value
        mid = (
            np.power(-1, cuts + 1)
            * coefficient
            * _get_sub_expectation_values(experiment_run, observables, shots)
        )
        expectation_values += mid

    # multiply by gamma to the power of cuts and take mean
    return np.power(4, cuts) * expectation_values / (samples)


def _get_sub_expectation_values(
    experiment_run: TotalResult, observables: list[int | list[int]], shots: int
) -> list:
    """Calculate sub expectation value for the result.

    Args:
    ----
        experiment_run: results of a subcircuit pair
        observables: list of observables as qubit indices (Z-observables)
        shots: number of shots

    Returns:
    -------
        list: list of sub expectation values

    """
    # generate all possible combinations between end of circuit measurements
    # from subcircuit group
    sub_circuit_result_combinations = product(*experiment_run.subcircuits[0])

    # initialize sub solution array
    sub_expectation_value = np.zeros(len(observables))
    for circuit_result in sub_circuit_result_combinations:  # loop through results
        # concat results to one array and reverse to account for qiskit quibit ordering
        full_result = np.concatenate(
            [i.measurements[0] for i in reversed(circuit_result)]
        )

        qpd_measurement_coefficient = 1  # initial value for qpd coefficient
        weight = shots  # initial weight
        for res in circuit_result:  # calculate weight and qpd coefficient
            weight *= res.count / shots
            if len(res.measurements) > 1:
                qpd_measurement_coefficient *= np.prod(res.measurements[1])
        observable_results = np.empty(len(observables))  # initialize empty array
        # for obsrvables
        for count, obs in enumerate(observables):  # populate observable array
            if isinstance(obs, int):
                observable_results[count] = full_result[obs]  # if single qubit
            # observable just save
            # to array
            else:  # if multi qubit observable
                multi_qubit_observable_eigenvalue = 1  # initial eigenvalue
                for sub_observables in obs:  # multio qubit observable
                    multi_qubit_observable_eigenvalue *= full_result[sub_observables]
                    observable_results[count] = (
                        np.power(-1, len(obs) + 1) * multi_qubit_observable_eigenvalue
                    )

        observable_expectation_value = (
            qpd_measurement_coefficient * observable_results * weight
        )
        sub_expectation_value += observable_expectation_value

    return sub_expectation_value


def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
) -> tuple[list[CutLocation], list[QuantumCircuit]]:
    """Get cut locations and subcircuits with placeholder operations.

    Args:
    -----
        circuit: circuit with cuts inserted

    Returns:
    --------
        cut_locations: a list of cut locations
        subcircuits: subcircuits with placeholder operations

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
    -----
        subcircuits: subcircuits containing the placeholder operations
        cut_locations: list of cut locations
        observables: list of observables as qubit indices (Z observable)
        backend: backend to use for running experiment circuits (optional)
        mitigate: wether or not to use readout error mitigation (optional)

    Returns:
    --------
        list: a list of expectation values

    """
    subexperiments, coefs, id_meas = get_experiment_circuits(subcircuits, cut_locations)
    if backend is not AerSimulator():
        subexperiments = transpile_experiments(subexperiments, backend)
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
    -----
        circuit: circuit with cut experiments
        observables: list of observbles in the form of qubit indices (Z-obsevable).
        backend: backend to use for running experiment circuits (optional)
        mitigate: wether or not to use readout error mitigation (optional)

    Returns:
    --------
        list: a list of expectation values

    """
    circuit = circuit.copy()
    qss, circs = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(circs, qss, observables, backend, mitigate)
