"""Circuit knitting wire cut functionality."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitError, CircuitInstruction, Qubit
from qiskit.quantum_info import PauliList
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator
from qiskit_experiments.library import LocalReadoutError

from .backend_utility import transpile_experiments
from .identity_qpd import identity_qpd

ERROR = 0.0000001

class QCutError(Exception):
    """Exception raised for custom error conditions.

    Attributes
    ----------
        message (str): Explanation of the error.
        code (int, optional): Error code representing the error type.

    """

    def __init__(self, message: str = "An error occurred", code: int | None = None) -> None:
        """Init.

        Args:
        ----
            message: Explanation of the error. Default is "An error occurred".
            code: Optional error code representing the error type.

        Attributes:
        ----------
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


#Class for storing results from single sub-circuit run

class SubResult:
    """Storage class for easier storage/access to the results of a subcircuit."""

    def __init__(self, measurements: list, count: int) -> None:
        """Init."""
        self.measurements = measurements #measurement results
        self.count = count #counts for this specific measurement

    def __str__(self) -> str:
        """Format string."""
        return f"{self.measurements}, {self.count}"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


#Store total results of all sub-circuits (two for now)
class TotalResult:
    """Storage class for easier access to the results of a subcircuit group."""

    def __init__(self, *subcircuits: list) -> None:
        """Init."""
        #sub1 and sub2 are SubResults
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

    def __init__(self, cut_location: tuple) -> None:
        """Init."""
        self.qubits = cut_location[0]
        self.meas = cut_location[0][0][1]
        self.init = cut_location[0][1][1]
        self.index = cut_location[1]

    def __eq__(self, other: CutLocation) -> bool:
        """Equality."""
        if not isinstance(other, CutLocation):
            return NotImplemented

        return self.meas == other.meas and self.init == other.init and self.index == other.index

    def __str__(self) -> str:
        """Format string."""
        return f"meas qubit: {self.meas}, init qubit: {self.init}, cut index: {self.index}"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)

def get_cut_locations(circuit: QuantumCircuit) -> list[CutLocation]:
    """Get the locations of the cuts in the circuit.

    Note:
    ----
        get_cut_location modifies the circuit passed as an argument by removing cut_wire gates. Therefore trying to run
        it multiple times in a row on the same circuit will fail. To avoid this it is not recommended to run
        this function by itself. Instead use get_locations_and_subcircuits().

    Args:
    ----
        circuit: Quantum circuit with Move() operations.

    Returns:
    -------
        Locations of the cuts as a list.

    """
    index = 0 #index of the current instruction in circuit_data
    circuit_data = circuit.data
    cut_locations = []
    offset = 0 #offset to account for removing the move operations

    #loop through circuit instructions
    #if operation is a Move() instruction remove it and add registers and
    # offset index to cut_locations

    #rename varibales to be more descriptive (namely qs)

    while index < len(circuit):
        if circuit_data[index].operation.name == "Cut":
            qs = (circuit.find_bit(x).registers[0] for x in circuit_data[index].qubits)
            circuit_data.remove(circuit_data[index])
            cut_locations.append(CutLocation((tuple(qs), index + offset)))
            index -= 1
        index += 1
    if len(cut_locations) == 0:
        exc = """No cuts in circuit. Did you pass the wrong circuit or try to run get_cut_location()
                multiple times in a row?"""
        raise QCutError(exc)
    return cut_locations

def get_bounds(cut_locations: list[CutLocation]) -> list:
    """Get the bounds for subcircuits as qubit indices.

    Args:
    ----
        cut_locations: Locations of the cuts as a list.

    Returns:
    -------
        Bounds as a list.

    """
    test = []
    for i, cut in enumerate(cut_locations):
        if i == 0:
            test.append(([cut.meas, cut.init], [cut.index], [[cut.meas, cut.init]]))
        for j in test:
            if cut.index in j[1]:
                j[0][0] = max(j[0][0], cut.meas)
                j[0][1] = max(j[0][1], cut.init)
                j[2].append([cut.meas, cut.init])
                break
            elif min(cut.meas, cut.init) - max(j[0]) < 0:
                j[0][0] = min(j[0][0], cut.meas, cut.init) if j[0][0] == min(j[0]) else max(j[0][0], cut.meas, cut.init)
                j[0][1] = min(j[0][1], cut.meas, cut.init) if j[0][1] == min(j[0]) else max(j[0][1], cut.meas, cut.init)
                j[1].append(cut.index)
                j[2].append([cut.meas, cut.init])
                break
        else:
            test.append(([cut.meas, cut.init], [cut.index], [[cut.meas, cut.init]]))

    bounds = []
    for i in test:
        bounds.append(max([min(x) for x in i[2]]))  # noqa: PERF401

    return bounds

def get_locations_and_bounds(circuit: QuantumCircuit) -> tuple[list, list]:
    """Get the locations of the cuts in the circuit and the subcircuit bounds.

    Args:
    ----
        circuit: Quantum circuit with Move() operations.

    Returns:
    -------
        Locations of the cuts and bounds as a list.

    """
    cut_locations = get_cut_locations(circuit)
    bounds = get_bounds(cut_locations)

    return cut_locations, bounds

#Placeholder measure node operation
c = QuantumCircuit(1, name="Meas")
measure_node = c.to_instruction()

#Placeholder initialize node operation
c = QuantumCircuit(1, name="Init")
initialize_node = c.to_instruction()

#Insert placeholders to the cut locations
def insert_meassure_prepare_channel(circuit: QuantumCircuit, cut_locations: list[CutLocation]) -> QuantumCircuit:
    """Insert the measure and initialize node at the cut locations.

    Args:
    ----
        circuit: Quantum circuit with Move() operations.
        cut_locations: Locations of the cuts as a list.

    Returns:
    -------
        circuit with measure and initialize nodes inserted
    Raises:

    """
    # make the varibales with deep arrays clearer. Maybe just another vaeriable with a clear name etc?
    # Or custom class

    circuit_data = circuit.data
    offset = 0
    for ind, i in enumerate(cut_locations):
        #Placeholder measure node operation
        c = QuantumCircuit(1, name=f"Meas_{ind}")
        measure_node = c.to_instruction()

        #Placeholder initialize node operation
        c = QuantumCircuit(1, name=f"Init_{ind}")
        initialize_node = c.to_instruction()
        if max(i.meas, i.init) == i.meas:
            circuit_data.insert(i.index+offset , CircuitInstruction(operation=measure_node,
                                                                 qubits=[Qubit(i.qubits[0][0], i.qubits[0][1])]))

            circuit_data.insert(i.index+offset , CircuitInstruction(operation=initialize_node,
                                                                 qubits=[Qubit(i.qubits[1][0], i.qubits[1][1])]))
            offset += 2
        else:
            circuit_data.insert(i.index+offset , CircuitInstruction(operation=initialize_node,
                                                                 qubits=[Qubit(i.qubits[1][0], i.qubits[1][1])]))

            circuit_data.insert(i.index+offset , CircuitInstruction(operation=measure_node,
                                                                 qubits=[Qubit(i.qubits[0][0], i.qubits[0][1])]))
            offset += 2
    return circuit.decompose(gates_to_decompose="decomp")

def get_locations_and_bounds_and_insert_meassure_prepare_channel(
        circuit: QuantumCircuit,
        ) -> tuple[list, list, QuantumCircuit]:
    """Get locations of cuts and the bounds for the circuit. Insert the measure and initialize nodes.

    Args:
    ----
        circuit: Quantum circuit with Move() operations.

    Returns:
    -------
        qss: Locations of the cuts as a list
        bounds: Subcircuit bounds as a list
        circ: Circuit with measure and initialize nodes inserted
    Raises:

    """
    circuit = circuit.copy() #make a copy of the circuit to avoid modifying the original one
    qss, bounds = get_locations_and_bounds(circuit)
    circ = insert_meassure_prepare_channel(circuit, qss)
    return qss, bounds, circ

def build_subcircuit(current_bound: int,
                     previous_bound: int,
                     clbits: int,
                     subcircuit_operations: list,
                     circuit: QuantumCircuit) -> QuantumCircuit:
    """Help build subcircuits.

    Args:
    ----
        circuit: Quantum circuit with Move() operations.
        previous_bound: last qubit of previous subcircuit.
        current_bound: last qubit of current subcircuit.
        clbits: number of classical bits.
        subcircuit_operations: operations for subcircuit to be built.

    Returns:
    -------
        circs: Array of subcircuits
    Raises:

    """
    #define quantum register
    qr = QuantumRegister(current_bound - previous_bound + 1)

    #define classical registers for obsrvable - and qpd - measurements
    crqpd = ClassicalRegister(clbits, "qpd")
    cr = ClassicalRegister(current_bound - previous_bound + 1 - clbits, "meas")

    subcircuit = QuantumCircuit(qr, crqpd, cr) #initialize the subcircuit
    for operation in subcircuit_operations: #loop throgh the subcircuitOperatiosn and add them to the subcircuit
        #get the qubits needed fot the operation and bind the to the quantum register of the subcircuit
        qubits_for_operation = tuple(Qubit(subcircuit.qregs[0],
                                            circuit.find_bit(qubit).index-previous_bound)
                                            for qubit in operation.qubits)

        # insert operation to subcricuit
        subcircuit.data.insert(len(subcircuit), CircuitInstruction(operation=operation.operation,
                                                                   qubits=qubits_for_operation))
    return subcircuit

#Cuts the given circuit into two at the location of the cut marker
def separate_sub_circuits(circuit: QuantumCircuit, sub_circuit_qubit_bounds: list) -> list[QuantumCircuit]:
    """Split the circuit with measure and prepare channels into separate subcircuits.

    Args:
    ----
        circuit: Quantum circuit with Move() operations.
        sub_circuit_qubit_bounds: Bounds for subcircuits as list of qubit indices.

    Returns:
    -------
        circs: Array of subcircuits
    Raises:

    """
    #remove barriers
    rb = RemoveBarriers()
    circuit = rb(circuit)

    #append final bound
    sub_circuit_qubit_bounds.append(circuit.num_qubits)
    subcircuits_list = [] #initialize solution array
    current_subcircuit = 0  #counter for which subcircuit we are in
    clbits = 0 #number of classical bits needed for subcircuit
    offset = 0 #offset from inserting operations
    previous_bound = 0 #previous bound
    subcircuit_operations = [] #array for collecting subcircuit operations

    for i,op in enumerate(circuit.data):

        qubits = [circuit.find_bit(x).index for x in op.qubits] #qubits in operations
        if "Meas" in op.operation.name:
            clbits += 1 #if measure node add a classical bit

        if i == len(circuit.data)-1: #if at the end of the original circuit

            #see if this can be refactored to be nicer

            subcircuit_operations.append(op) #append the final operation to list

            #define quantum register
            qr = QuantumRegister(sub_circuit_qubit_bounds[current_subcircuit] - previous_bound)

            #define classical registers for obsrvable - and qpd - measurement
            crqpd = ClassicalRegister(clbits, "qpd")
            cr = ClassicalRegister(sub_circuit_qubit_bounds[current_subcircuit] - previous_bound - clbits, "meas")

            subcircuit = QuantumCircuit(qr, crqpd, cr) #initialize the subcircuit

            for operation in subcircuit_operations: #loop throgh the subcircuitOperatiosn and add them to the subcircuit

                #get the qubits needed fot the operation and bind the to the quantum register of the subcircuit
                qubits_for_operation = tuple(Qubit(subcircuit.qregs[0], circuit.find_bit(x).index-offset)
                                             for x in operation.qubits)

                #insert operation to subcricuit
                subcircuit.data.insert(len(subcircuit), CircuitInstruction(operation=operation.operation,
                                                                           qubits=qubits_for_operation))

            subcircuits_list.append(subcircuit)
            continue

        #if sub_circuit_qubit_bounds[current_subcircuit]+1 in qubits: #if at the end of subcircuit
        if True in (x>sub_circuit_qubit_bounds[current_subcircuit] for x in qubits):
            #build the subcircuit
            subcircuit = build_subcircuit(sub_circuit_qubit_bounds[current_subcircuit] , previous_bound, clbits,
                                          subcircuit_operations, circuit)
            subcircuits_list.append(subcircuit)

            #reset variables
            subcircuit_operations = []
            clbits = 0
            previous_bound = sub_circuit_qubit_bounds[current_subcircuit] + 1
            current_subcircuit += 1
            if current_subcircuit < len(sub_circuit_qubit_bounds):
                offset = sub_circuit_qubit_bounds[current_subcircuit]
        subcircuit_operations.append(op)
    return subcircuits_list

def get_qpd_combinations(cut_locations: list[CutLocation]) -> list:
    """Get all possible combinations of the QPD operations so that each combination has len(cutLocations) elements.

    Args:
    ----
        cut_locations: cut locations
    Returns:
        list of the possible QPD operations
    Raises:

    """
    return list(product(identity_qpd,repeat=len(cut_locations)))

def get_experiment_circuits(subcircuits: list[QuantumCircuit], # noqa: C901, PLR0912, PLR0915
                            cut_locations: list[CutLocation]) -> tuple[list, list, list]:
    """Generate all possible experiment circuits by inserting QPD operations on measure/initialize nodes.

    Args:
    ----
        subcircuits: subcircuits with measure/initialize nodes.
        cut_locations: cut locations.

    Returns:
    -------
        experimentCircuits: list of experiment circuits.
        coefficients: sign coefficients for each circuit.
        id_meas: list of index pointers to results that need additional post-processing due to
                 identity basis measurement.

    """
    qpd_combinations = get_qpd_combinations(cut_locations) #generate the QPD operation combinations

    #initialize solution lists
    cuts = len(cut_locations)
    num_circs = np.power(8,cuts)
    experiment_circuits = []
    id_meas = np.full((num_circs, 3), None)
    num_id_meas = 0
    coefficients = np.empty(num_circs)

    for id_meas_eperiment_index, qpd in enumerate(qpd_combinations): #loop through all QPD combinations
        coefficient = np.prod([op["c"] for op in qpd])
        coefficients[id_meas_eperiment_index] = np.prod(coefficient)
        sub_experiment_circuits = [] #sub array for collecting related experiment circuits

        inserted_operations = 0
        for id_meas_subcircuit_index, circ in enumerate(subcircuits):
            subcircuit = deepcopy(circ)

            classical_bit_index = 0
            id_meas_bit = 0
            qpd_qubits = [] #store the qubit indices of qubits used for qpd measurements
            for op_ind, op in enumerate(subcircuit.data):
                if inserted_operations >= 2 * cuts: #if looped through all cuts, stop
                    break
                if "Meas" in op.operation.name: #if measure channel remove placeholder and insert current
                                               # qpd operation
                    qubit_index = subcircuit.find_bit(op.qubits[0]).index
                    subcircuit.data.pop(op_ind) #remove plaxceholder measure channel
                    qpd_qubits.append(qubit_index) #store index
                    qubits_for_operation = tuple(Qubit(subcircuit.qregs[0], qubit_index) for x in op.qubits)
                    if qpd[int(op.operation.name.split("_")[-1])]["op"].name == "id-meas": #if identity measure channel

                        #store indices
                        id_meas[num_id_meas] = np.array([id_meas_eperiment_index, id_meas_subcircuit_index,
                                                         id_meas_bit])
                        num_id_meas += 1
                        #remove extra classical bits and registers
                        if len(subcircuit.cregs) > 1:
                            if subcircuit.cregs[0].size == 1:
                                del subcircuit.clbits[subcircuit.cregs[0].size-1]
                                del subcircuit.cregs[0]._bits[subcircuit.cregs[0].size-1]  # noqa: SLF001
                                del subcircuit.cregs[0]
                            else:
                                del subcircuit.clbits[subcircuit.cregs[0].size-1]
                                del subcircuit.cregs[0]._bits[subcircuit.cregs[0].size-1]  # noqa: SLF001
                                subcircuit.cregs[0]._size -= 1  # noqa: SLF001

                        subcircuit.data.insert(op_ind, CircuitInstruction(operation=
                                                                       qpd[int(op.operation.name.split("_")[-1])]["op"],
                                                                       qubits=qubits_for_operation))
                    else:
                        subcircuit.data.insert(op_ind, CircuitInstruction(operation=
                                                                       qpd[int(op.operation.name.split("_")[-1])]["op"],
                                                                       qubits=qubits_for_operation,
                                                                       clbits=[subcircuit.cregs[0][classical_bit_index]]))

                        #increment classical bit counter
                        classical_bit_index += 1

                    id_meas_bit += 1
                    inserted_operations += 1

                if "Init" in op.operation.name:
                    subcircuit.data.pop(op_ind)
                    qubits_for_operation = tuple(Qubit(subcircuit.qregs[0],
                                                       subcircuit.find_bit(x).index) for x in op.qubits)
                    subcircuit.data.insert(op_ind,CircuitInstruction(
                                                        operation=qpd[int(op.operation.name.split("_")[-1])]["init"],
                                                        qubits=qubits_for_operation))

                    inserted_operations += 1

            meas_qubits = [x for x in range(subcircuit.num_qubits) if x not in qpd_qubits]
            if len(subcircuit.cregs) >= 2:  # noqa: PLR2004
                subcircuit.measure(meas_qubits, subcircuit.cregs[1])
            else:
                subcircuit.measure(meas_qubits, subcircuit.cregs[0])

            decomp = ["z-meas", "y-meas", "x-meas", "id-meas", "1-init", "0-init",
                      "'+'-init", "'-'-init", "'i+'-init", "'i-'-init"]
            subcircuit = subcircuit.decompose(gates_to_decompose=decomp)
            sub_experiment_circuits.append(subcircuit)
        experiment_circuits.append(sub_experiment_circuits)

    return experiment_circuits, coefficients, id_meas[:num_id_meas]

def run_experiments(experiment_circuits: list,  # noqa: PLR0913
                    cut_locations: list[CutLocation],
                    id_meas: list,
                    shots: int = 2**12,
                    backend: int | None = None,
                    mitigate: bool = False) -> list:  # noqa: FBT001, FBT002
    """Run experiment circuits.

    Args:
    ----
        experiment_circuits: experiment circuits
        cut_locations: list of cut locations
        id_meas: list of identity basis measurement locations
        shots: number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)
        mitigate: wether to use readout error mitigation or not (optional)

    Returns:
    -------
        processed_results: list of transformed results

    """
    cuts = len(cut_locations)
    #number of samples neede
    samples = int(np.power(4, (2)*cuts)/np.power(ERROR,2))
    samples = int(samples / len(experiment_circuits))
    if backend is None:
        backend = AerSimulator()

    results = [0]*(len(experiment_circuits))

    for count, subcircuit_group in enumerate(experiment_circuits):
        sub_result = [backend.run(i, shots=shots).result().get_counts() for i in subcircuit_group]
        if mitigate:
            qss = set()
            for res in sub_result:
                q = list(res.keys())
                qs = len(q[0].replace(" ", ""))
                qss.add(qs)

            mitigators = {qs: LocalReadoutError(list(range(qs))).run(backend)
                          .analysis_results("Local Readout Mitigator").value
                          for qs in qss}

            for ind, res in enumerate(sub_result):
                q = list(res.keys())
                qs = list(range(len(q[0].replace(" ", ""))))
                mitigator = mitigators[len(qs)]
                meas_bits = len(q[0].split(" ")[0])
                mitigated_quasi_probs = mitigator.quasi_probabilities(res)
                probs_test = {f"{int(old_key):0{len(qs)}b}"[::-1][:meas_bits] + " " +
                              f"{int(old_key):0{len(qs)}b}"[::-1][meas_bits:]:
                              mitigated_quasi_probs[old_key]*shots if mitigated_quasi_probs[old_key] > 0 else 0
                                    for old_key in mitigated_quasi_probs}
                sub_result[ind] = probs_test

        results[count] = sub_result

        sub_result = []
    return process_results(results, id_meas, shots, samples)


def process_results(results: list, id_meas: list, shots: int, samples: int) -> list:
    """Transform results with post processing function {0,1} -> [-1, 1].

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
                #separate end measurements from mid-circuit measurements
                #for running on Helmi need to change the circuit construction to avoid
                #mid circuit measurements. Possible for single / parallel wire cuts.
                separate_measurements = meassurements.split(" ")

                #map to eigenvalues
                result_eigenvalues = [np.array([-1 if x == "0" else 1 for x in i]) for i in separate_measurements]
                circuit_results.append(SubResult(result_eigenvalues, count/shots*samples))
            experiment_run_results.append(circuit_results)
        preocessed_results.append(TotalResult(experiment_run_results))

    for loc in id_meas:
        for i in preocessed_results[loc[0]].subcircuits[0][loc[1]]:
            if len(i.measurements) == 1:
                i.measurements.append(np.array([-1]))
            else:
                i.measurements[1] = np.insert(i.measurements[1], loc[2], -1)
    return preocessed_results

#Calculate the approx expectation values for the original circuit
#Soon hopefully more than Z-observables
def estimate_expectation_values(results: list[TotalResult],
                                coefficients: list[int],
                                cut_locations: list[CutLocation],
                                observables: list) -> list:
    """Calculate the estimated expectation values.

    Args:
    ----
        results: results from experiment circuits
        coefficients: list of coefficients for each subcircuit group
        cut_locations: cut locations
        observables: observables to calculate expectation values for

    Returns:
    -------
        list: expectations as a list of floats list of floats

    """
    cuts = len(cut_locations)
    #number of samples neede
    samples = int(np.power(4, 2*cuts)/np.power(ERROR,2))
    shots = int(samples / len(results))

    #ininialize approx expectation values of an array of ones
    #could also be zeros don't think it really matters
    expectation_values = np.ones(len(observables))
    for experiment_run, coefficient in zip(results, coefficients):
        #add sub results to the total approx expectation value
        mid = (np.power(-1, cuts+1) * coefficient*
               get_sub_expectation_values(experiment_run, observables, shots))
        expectation_values += mid

    #multiply by gamma to the power of cuts and take mean
    return np.power(4,cuts) * expectation_values / (samples)

def get_sub_expectation_values(experiment_run: TotalResult, observables: list, shots: int) -> list:
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
    sub_circuit_result_combinations = product(*experiment_run.subcircuits[0])

    sub_expectation_value = np.zeros(len(observables))
    total_weight = 0
    for circuit_result in sub_circuit_result_combinations:

        full_result = np.concatenate([i.measurements[0] for i in reversed(circuit_result)])

        qpd_measurement_coefficient = 1
        weight = shots
        for res in circuit_result:
            weight *= res.count/shots
            if len(res.measurements) > 1:
                qpd_measurement_coefficient *= np.prod(res.measurements[1])
        total_weight += weight
        observable_results  = np.empty(len(observables))
        for count, obs in enumerate(observables):
            if isinstance(obs, int):
                observable_results[count] = full_result[obs]
            else:
                multi_qubit_observable_eigenvalue = 1
                for sub_observables in obs:
                    multi_qubit_observable_eigenvalue *= full_result[sub_observables]
                    observable_results[count] = (np.power(-1, len(obs)+1)
                                                 *multi_qubit_observable_eigenvalue)
        observable_expectation_value = qpd_measurement_coefficient*observable_results*weight
        sub_expectation_value += observable_expectation_value

    return sub_expectation_value

def get_locations_and_subcircuits(circuit: QuantumCircuit) -> tuple[list, list]:
    """Get cut locations and subcircuits with placeholder operations.

    Args:
    ----
        circuit: circuit with cuts inserted

    Returns:
    -------
        cut_locations: a list of cut locations
        subcircuits: subcircuits with placeholder operations

    """
    circuit = circuit.copy()
    cut_locations = get_cut_locations(circuit)
    sorted_cut_locations = sorted(cut_locations, key = lambda x: min(x.meas, x.init))
    circ = insert_meassure_prepare_channel(circuit, cut_locations)
    bounds = get_bounds(sorted_cut_locations)
    try:
        subcircuits = separate_sub_circuits(circ, bounds)
    except CircuitError as e:
        msg = "Invalid cut placement. See documentation for how cuts should be placed."
        raise QCutError(msg) from e

    return sorted_cut_locations, subcircuits

def run_cut_circuit(subcircuits: list,
                    cut_locations: list,
                    observables: list,
                    backend=AerSimulator(),  # noqa: ANN001, B008
                    mitigate: bool = False) -> list:  # noqa: FBT001, FBT002
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
    ----
        subcircuits: subcircuits containing the placeholder operations
        cut_locations: list of cut locations
        observables: list of observables as qubit indices (Z observable)
        backend: backend to use for running experiment circuits (optional)
        mitigate: wether or not to use readout error mitigation (optional)

    Returns:
    -------
        list: a list of expectation values

    """
    subexperiments, coefs, id_meas = get_experiment_circuits(subcircuits, cut_locations)
    if backend is not AerSimulator():
        subexperiments = transpile_experiments(subexperiments, backend)
    results = run_experiments(subexperiments, cut_locations, id_meas=id_meas,
                              backend=backend, mitigate=mitigate)

    return estimate_expectation_values(results, coefs, cut_locations, observables)

def get_pauli_list(input_list: list, length: int) -> PauliList:
    """Transform list of observable indices to Paulilist of Z observables.

    Args:
    ----
        input_list: lits of observables as qubit indices
        length: number of qubits in the circuit

    Returns:
    -------
        PauliList: a PauliList of Z observables

    """
    result = []
    base_string = "I" * length

    for indices in input_list:
        temp_string = list(base_string)
        if isinstance(indices, int):
            temp_string[indices] = "Z"
        else:
            for index in indices:
                temp_string[index] = "Z"
        result.append("".join(temp_string))

    return PauliList(result)

def run(circuit: QuantumCircuit,
        observables: list,
        backend = AerSimulator(),  # noqa: ANN001, B008
        mitigate: bool = False) -> list:  # noqa: FBT001, FBT002
    """Run the whole circuit knitting sequence with one function call.

    Args:
    ----
        circuit: circuit with cut experiments
        observables: list of observbles in the form of qubit indices (Z-obsevable).
        backend: backend to use for running experiment circuits (optional)
        mitigate: wether or not to use readout error mitigation (optional)

    Returns:
    -------
        list: a list of expectation values

    """
    circuit = circuit.copy()
    qss, circs = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(circs, qss, observables, backend, mitigate)

