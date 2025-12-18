"""Circuit knitting wire cut functionality."""

from __future__ import annotations

import pickle
from itertools import product
from typing import TYPE_CHECKING, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.converters import circuit_to_dag
from qiskit_aer import AerSimulator

from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.qcutresult import SubResult, TotalResult
from QCut.qpd import cz_qpd, identity_qpd

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate, Instruction
from qiskit.circuit.library import CXGate, ECRGate, HGate, SdgGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Operator, pauli_basis
 
import numpy as np
 
from typing import Iterable, Optional

if TYPE_CHECKING:
    from collections.abc import Iterable

ERROR = 0.0000001


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
        cut_locations (np.ndarray[CutLocation]): cut locations

    Returns:
        Iterable[tuple[dict]]:
            Iterable of the possible QPD operations

    """
    qpd_lists = []
    for cut in cut_locations:
        if isinstance(cut, SingleQubitCutLocation):
            qpd_lists.append(identity_qpd)
        elif isinstance(cut, CutLocation):
            qpd_lists.append(cz_qpd)
        else:
            raise TypeError(f"Unknown cut type: {type(cut)}")

    # Cartesian product across all qpd options per cut
    all_combinations = product(*qpd_lists)
    return all_combinations


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

    dag = circuit_to_dag(subcircuit)
    idle = list(dag.idle_wires())

    for wire in idle:
        if isinstance(wire, Qubit) and wire._index in meas_qubits:
            meas_qubits.remove(wire._index)

    if len(subcircuit.cregs) >= 2:
        subcircuit.measure(meas_qubits, subcircuit.cregs[1])
    else:
        subcircuit.measure(meas_qubits, subcircuit.cregs[0])
    return subcircuit


def get_placeholder_locations(subcircuits: list[QuantumCircuit]) -> list:
    """
    Identify the locations of placeholder operations in a list of quantum subcircuits.
    This function scans through each quantum subcircuit provided in the input list and
    identifies the indices and operations where either measurement ("Meas") or
    initialization ("Init") operations occur. It returns a list of lists, where each
    sublist corresponds to  a subcircuit and contains tuples of the
    form (index, operation).

    Args:
        subcircuits (list[QuantumCircuit]):
            A list of QuantumCircuit objects to be analyzed.
    Returns:
        list:
            A list of lists, where each sublist contains tuples (index, operation)
            indicating the positions of measurement or initialization operations in
            the corresponding subcircuit.

    """
    ops = []
    names = ["Meas", "Init", "cutCZ"]
    for circ in subcircuits:
        subops = []
        for ind, op in enumerate(circ):
            # if "Meas" in op.operation.name or "Init" in op.operation.name :
            if any(i in op.operation.name for i in names):
                subops.append((ind, op))
        ops.append(subops)

    return ops


def _remove_obsm(subcircuits: list[dict[int, QuantumCircuit]]) -> list[dict[int, QuantumCircuit]]:

    for obs_set in subcircuits:
        for ind, circ in obs_set.items():
            j = 0
            while j < len(circ.data):
                if "obs" in circ[j].operation.name:
                    circ.data.remove(circ[j])
                else:
                    j += 1

def _remove_obsm_old(subcircuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
    for i in subcircuits:
        j = 0
        while j < len(i.data):
            if "obs" in i[j].operation.name:
                i.data.remove(i[j])
            else:
                j += 1

def insert_wire_cut_qpd(
    ind,
    op,
    subcircuit,
    offset,
    qpd_qubits,
    qpd,
    id_meas,
    num_id_meas,
    id_meas_experiment_index,
    id_meas_subcircuit_index,
    id_meas_bit,
    classical_bit_index,
    inserted_operations,
):
    if "Meas" in op.operation.name:  # if measure channel remove placeholder
        # and insert current
        # qpd operation
        qubit_index = subcircuit.find_bit(op.qubits[0]).index
        subcircuit.data.pop(ind + offset)  # remove plaxceholder
        # measure channel
        qpd_qubits.append(qubit_index)  # store index
        qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
        meas_op = qpd[int(op.operation.name.split("_")[-1])]["op_0"]
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
                    CircuitInstruction(operation=subop.operation, 
                                       qubits=qubits_for_operation),
                )
        else:
            for i, subop in enumerate(reversed(meas_op.data)):
                if i == 0:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                            clbits=[subcircuit.cregs[0][classical_bit_index]],
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
        offset += len(meas_op.data) - 1

    if "Init" in op.operation.name:
        subcircuit.data.pop(ind + offset)
        init_op = qpd[int(op.operation.name.split("_")[-1])]["op_1"]
        qubits_for_operation = [
            Qubit(subcircuit.qregs[0], subcircuit.find_bit(x).index) for x in op.qubits
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

    return offset, num_id_meas, id_meas_bit, classical_bit_index, inserted_operations


def insert_cz_cut_qpd(  # noqa: C901
    ind,
    op,
    subcircuit,
    offset,
    qpd_qubits,
    qpd,
    id_meas,
    num_id_meas,
    id_meas_experiment_index,
    id_meas_subcircuit_index,
    id_meas_bit,
    classical_bit_index,
    inserted_operations,
):
    if "_c_" in op.operation.name:  # if measure channel remove placeholder
        # and insert current
        # qpd operation

        qubit_index = subcircuit.find_bit(op.qubits[0]).index
        subcircuit.data.pop(ind + offset)  # remove plaxceholder
        # measure channel
        # qpd_qubits.append(qubit_index)  # store index
        qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
        meas_op = qpd[int(op.operation.name.split("_")[-1])]["op_0"]
        if meas_op.name in ["id-meas", "s", "sdg", "z"]:
            # if identity measure channel
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
                        operation=subop.operation, qubits=qubits_for_operation
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
                            clbits=[subcircuit.cregs[0][classical_bit_index]],
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

    if "_t_" in op.operation.name:
        # and insert current
        # qpd operation
        qubit_index = subcircuit.find_bit(op.qubits[0]).index
        subcircuit.data.pop(ind + offset)  # remove plaxceholder
        # measure channel
        # qpd_qubits.append(qubit_index)  # store index
        qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
        meas_op = qpd[int(op.operation.name.split("_")[-1])]["op_1"]
        if meas_op.name in ["id-meas", "s", "sdg", "z"]:
            # if identity measure channel
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
                        operation=subop.operation, qubits=qubits_for_operation
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
                            clbits=[subcircuit.cregs[0][classical_bit_index]],
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

    return offset, num_id_meas, id_meas_bit, classical_bit_index, inserted_operations


def get_experiment_circuits_old(  # noqa: C901
    subcircuits: list[QuantumCircuit] | CutCircuit,  # noqa: C901
    cut_locations: np.ndarray[CutLocation],
) -> tuple[CutCircuit, list[int], list[tuple[int, int, int]]]:
    """Generate experiment circuits by inserting QPD operations on
    measure/initialize nodes.

    Loop through qpd combinations. Calculate coefficient for subcircuit group by
    taking the product of all coefficients in the current qpd row. Loop through
    subcircuits generated in 4. Make deepcopy of subcircuit and iterate over its
    circuit data. When hit either Meas_{ind} of Init_{ind} repace it with operation
    found in qpd[ind]["op_0"/"op_1"]. WHile generating experiment circuits also
    generate a list of locations that have an identity basis measurement. These
    measurement outcomes need to be added during post-processing. Locations added as
    [index of experiment circuit, index of subcircuit, index of classical bit
    corresponding to measurement]. Repeat untill looped through all qpd rows.
    sircuits reutrned as [circuit_group0, circuit_group1, ...], where circuit_goup
    is [subciruit0, subcircuit1, ...].

    Args:
        subcircuits (list[QuantumCircuit]): subcircuits with measure/initialize nodes.
        cut_locations (np.ndarray[CutLocation]): cut locations.

    Returns:
        tuple: A tuple containing:
            - CutCircuit: A CutCircuit object containing the experiment circuits.
            - list[int]: A list of coefficients for each circuit.
            - list[tuple[int, int, int]]:
                A list of index pointers to results that need additional post-processing
                due to identity basis measurement.

    """
    qpd_combinations = get_qpd_combinations(cut_locations)  # generate the QPD
    # operation combinations

    check_circuit_type = (isinstance(subcircuits, CutCircuit) 
                          and subcircuits.backend is not None)
    

    if check_circuit_type:
        backend = subcircuits.backend
        try:
            basis = backend.configuration().basis_gates
        except Exception:
            basis = list(backend.architecture.gates.keys())
        basis = ["r" if gate == "prx" else gate for gate in basis]
        
        subcircuits = subcircuits.subcircuits

    _remove_obsm(subcircuits)

    # initialize solution lists
    cuts = len(cut_locations)
    cz_cuts = len([i for i in cut_locations if isinstance(i, CutLocation)])
    wire_cuts = cuts - cz_cuts

    num_circs = np.power(8, wire_cuts) * np.power(6, cz_cuts)
    experiment_circuits = []
    num_id_meas_init = cuts * 2 * np.power(8, cuts - 1) * np.power(6, cz_cuts)
    id_meas = np.full((num_id_meas_init, 3), None)
    num_id_meas = 0
    coefficients = np.empty(num_circs)
    placeholder_locations = get_placeholder_locations(subcircuits)
    for id_meas_experiment_index, qpd in enumerate(
        qpd_combinations
    ):  # loop through all
        # QPD combinations
        coefficients[id_meas_experiment_index] = np.prod([op["c"] for op in qpd])

        if check_circuit_type:
            for sub in qpd:
                sub["op_0"] = transpile(sub["op_0"], basis_gates=basis)
                sub["op_1"] = transpile(sub["op_1"], basis_gates=basis)

        sub_experiment_circuits = []  # sub array for collecting related experiment
        # circuits
        inserted_operations = 0
        for id_meas_subcircuit_index, circ in enumerate(subcircuits):
            subcircuit = pickle.loads(pickle.dumps(circ))
            # subcircuit = deepcopy(circ)
            offset = 0
            classical_bit_index = 0
            id_meas_bit = 0
            qpd_qubits = []  # store the qubit indices of qubits used for qpd
            # measurements
            for op_ind in placeholder_locations[id_meas_subcircuit_index]:
                ind, op = op_ind
                if "cut" in op.operation.name:
                    (
                        offset,
                        num_id_meas,
                        id_meas_bit,
                        classical_bit_index,
                        inserted_operations,
                    ) = insert_cz_cut_qpd(
                        ind,
                        op,
                        subcircuit,
                        offset,
                        qpd_qubits,
                        qpd,
                        id_meas,
                        num_id_meas,
                        id_meas_experiment_index,
                        id_meas_subcircuit_index,
                        id_meas_bit,
                        classical_bit_index,
                        inserted_operations,
                    )

                else:
                    (
                        offset,
                        num_id_meas,
                        id_meas_bit,
                        classical_bit_index,
                        inserted_operations,
                    ) = insert_wire_cut_qpd(
                        ind,
                        op,
                        subcircuit,
                        offset,
                        qpd_qubits,
                        qpd,
                        id_meas,
                        num_id_meas,
                        id_meas_experiment_index,
                        id_meas_subcircuit_index,
                        id_meas_bit,
                        classical_bit_index,
                        inserted_operations,
                    )

            subcircuit = _finalize_subcircuit(subcircuit, qpd_qubits)
            sub_experiment_circuits.append(subcircuit)
        experiment_circuits.append(sub_experiment_circuits)
    return CutCircuit(experiment_circuits), coefficients, id_meas[:num_id_meas]

def get_needed_measurements_per_qubit(op: SparsePauliOp) -> dict[int, set[str]]:
    """Get the needed measurements per qubit for a given SparsePauliOp.

    Args:
        op (SparsePauliOp): The SparsePauliOp to analyze.
    Returns:
        dict[int, set[str]]: A dictionary mapping qubit indices to sets of needed measurements.
    """
    needed_measurements = {}
    for pauli_string in op.paulis:
        for qubit_index, pauli in enumerate(pauli_string.to_label()):
            if pauli != "I":
                if qubit_index not in needed_measurements:
                    needed_measurements[qubit_index] = set()
                needed_measurements[qubit_index].add(pauli)
    return needed_measurements


def combine_measurements(needed_measurements: dict[int, set[str]]) -> list[dict[int, str]]:
    """Combine measurements to minimize the number of measurement settings.

    Args:
        needed_measurements (dict[int, set[str]]): A dictionary mapping qubit indices to sets of needed measurements.
    Returns:
        list[dict[int, str]]: A list of measurement settings, each represented as a dictionary mapping qubit indices to measurements.
    """
    measurement_settings = []
    while needed_measurements:
        setting = {}
        for qubit_index in list(needed_measurements.keys()):
            paulis = needed_measurements[qubit_index]
            if paulis:
                pauli = paulis.pop()
                setting[qubit_index] = pauli
                if not paulis:
                    del needed_measurements[qubit_index]
        measurement_settings.append(setting)

    return measurement_settings

class ModifyMeasurementBasis(TransformationPass):
 
    def __init__(
        self,
        measurement_settings: list[dict[int, str]],
        ops: dict[str, Instruction] | None = None,

    ):
        
        self.measurement_settings = measurement_settings
        self.ops = ops
        super().__init__()
 
    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        # collect all nodes in DAG and proceed if it is to be twirled
        
        no_obs = True

        print("Modifying measurement basis for measurement settings:", self.measurement_settings)

        for node in dag.op_nodes():
            if "obs" not in node.op.name:
                continue
 
            obs_ind = int(node.op.name.split("_")[-1])
            
            for setting in self.measurement_settings:
                if obs_ind not in setting:
                    #continue
                    dag.remove_op_node(node)
                    break
                
                print("Modifying measurement basis for observation index:", obs_ind)
                ob = setting[obs_ind]
                no_obs = False
                print("Measurement basis:", ob)
                
                mini_dag = DAGCircuit()
                register = QuantumRegister(1)
                mini_dag.add_qreg(register)

                if ob == "X":
                    if self.ops and "X-meas" in self.ops:
                        mini_dag.apply_operation_back(
                            self.ops["X-meas"], [register[0]]
                        )
                    else:
                        mini_dag.apply_operation_back(
                            HGate(), [register[0]]
                        )
                elif ob == "Y":
                    if self.ops and "Y-meas" in self.ops:
                        mini_dag.apply_operation_back(
                            self.ops["Y-meas"], [register[0]]
                        )
                    else:
                        mini_dag.apply_operation_back(
                            SdgGate(), [register[0]]
                        )
                        mini_dag.apply_operation_back(
                            HGate(), [register[0]]
                        )
                
                dag.substitute_node_with_dag(node, mini_dag)

        if no_obs:
            print("No observation nodes found; returning empty DAGCircuit.")
            return DAGCircuit()
        return dag

def get_obs_subcircuits(subcircuits: list[QuantumCircuit], 
                        measurement_settings: list[dict[int, str]],
                        ops: dict[str, Instruction] | None = None
                        ) -> list[dict[int, QuantumCircuit]]:
    print("Generating observable subcircuits for measurement settings:", measurement_settings)
    pms = [PassManager([ModifyMeasurementBasis([setting], ops)]) for setting in measurement_settings]
    obs_subcircuits = []
    for pm in pms:
        pm_circs = {}
        for ind, subcircuit in enumerate(subcircuits):
            modified_circuit = pm.run(subcircuit)
            if modified_circuit.num_qubits == 0:
                continue
            pm_circs[ind] = modified_circuit
        obs_subcircuits.append(pm_circs)
    return obs_subcircuits

def get_experiment_circuits(  # noqa: C901
    cut_circuit: CutCircuit,
    observables: SparsePauliOp,
) -> CutExperiment:
    """Generate experiment circuits by inserting QPD operations on
    measure/initialize nodes.

    Loop through qpd combinations. Calculate coefficient for subcircuit group by
    taking the product of all coefficients in the current qpd row. Loop through
    subcircuits generated in 4. Make deepcopy of subcircuit and iterate over its
    circuit data. When hit either Meas_{ind} of Init_{ind} repace it with operation
    found in qpd[ind]["op_0"/"op_1"]. WHile generating experiment circuits also
    generate a list of locations that have an identity basis measurement. These
    measurement outcomes need to be added during post-processing. Locations added as
    [index of experiment circuit, index of subcircuit, index of classical bit
    corresponding to measurement]. Repeat untill looped through all qpd rows.
    sircuits reutrned as [circuit_group0, circuit_group1, ...], where circuit_goup
    is [subciruit0, subcircuit1, ...].

    Args:
        subcircuits (list[QuantumCircuit]): subcircuits with measure/initialize nodes.
        cut_locations (np.ndarray[CutLocation]): cut locations.

    Returns:
        tuple: A tuple containing:
            - CutCircuit: A CutCircuit object containing the experiment circuits.
            - list[int]: A list of coefficients for each circuit.
            - list[tuple[int, int, int]]:
                A list of index pointers to results that need additional post-processing
                due to identity basis measurement.

    """
    qpd_combinations = get_qpd_combinations(cut_circuit.cut_locations)  # generate the QPD
    # operation combinations

    check_circuit_type = cut_circuit.backend is not None
    
    measurement_settings = combine_measurements(get_needed_measurements_per_qubit(observables))

    print("Measurement settings:", measurement_settings)

    backend = None
    if check_circuit_type:
        backend = cut_circuit.backend
        try:
            basis = backend.configuration().basis_gates
        except Exception:
            basis = list(backend.architecture.gates.keys())
        basis = ["r" if gate == "prx" else gate for gate in basis]

    obs_subcircuits = None

    if check_circuit_type:
        x_meas_ops = QuantumCircuit(1)
        x_meas_ops.h(0)
        x_meas_ops.name = "X-meas"
        x_meas_ops = transpile(x_meas_ops, basis_gates=basis)
        x_meas_ops = x_meas_ops.to_instruction()

        y_meas_ops = QuantumCircuit(1)
        y_meas_ops.sdg(0)
        y_meas_ops.h(0)
        y_meas_ops.name = "Y-meas"
        y_meas_ops = transpile(y_meas_ops, basis_gates=basis)
        y_meas_ops = y_meas_ops.to_instruction()

        ops = {"X-meas": x_meas_ops, "Y-meas": y_meas_ops}

        obs_subcircuits = get_obs_subcircuits(
            cut_circuit.subcircuits, measurement_settings, ops
        )
    else:
        obs_subcircuits = get_obs_subcircuits(
            cut_circuit.subcircuits, measurement_settings
        )



    _remove_obsm(obs_subcircuits)

    # initialize solution lists
    cuts = len(cut_circuit.cut_locations)
    cz_cuts = len([i for i in cut_circuit.cut_locations if isinstance(i, CutLocation)])
    wire_cuts = cuts - cz_cuts

    num_circs = np.power(8, wire_cuts) * np.power(6, cz_cuts)
    experiment_circuits = []
    num_id_meas_init = cuts * 2 * np.power(8, cuts - 1) * np.power(6, cz_cuts)
    id_meas = np.full((num_id_meas_init, 3), None)
    num_id_meas = 0
    coefficients = np.empty(num_circs)
    placeholder_locations = get_placeholder_locations(cut_circuit.subcircuits)
    for id_meas_experiment_index, qpd in enumerate(
        qpd_combinations
    ):  # loop through all
        # QPD combinations
        coefficients[id_meas_experiment_index] = np.prod([op["c"] for op in qpd])

        if check_circuit_type:
            for sub in qpd:
                sub["op_0"] = transpile(sub["op_0"], basis_gates=basis)
                sub["op_1"] = transpile(sub["op_1"], basis_gates=basis)

        sub_experiment_circuits = {}  # sub array for collecting related experiment
        # circuits
        inserted_operations = 0
        obs_set_circuits = []
        for obs_set in obs_subcircuits:
            cur_set_circuits = {}
            for id_meas_subcircuit_index, circ in obs_set.items():
                subcircuit = pickle.loads(pickle.dumps(circ))
                # subcircuit = deepcopy(circ)
                offset = 0
                classical_bit_index = 0
                id_meas_bit = 0
                qpd_qubits = []  # store the qubit indices of qubits used for qpd
                # measurements
                for op_ind in placeholder_locations[id_meas_subcircuit_index]:
                    ind, op = op_ind
                    if "cut" in op.operation.name:
                        (
                            offset,
                            num_id_meas,
                            id_meas_bit,
                            classical_bit_index,
                            inserted_operations,
                        ) = insert_cz_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            id_meas,
                            num_id_meas,
                            id_meas_experiment_index,
                            id_meas_subcircuit_index,
                            id_meas_bit,
                            classical_bit_index,
                            inserted_operations,
                        )

                    else:
                        (
                            offset,
                            num_id_meas,
                            id_meas_bit,
                            classical_bit_index,
                            inserted_operations,
                        ) = insert_wire_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            id_meas,
                            num_id_meas,
                            id_meas_experiment_index,
                            id_meas_subcircuit_index,
                            id_meas_bit,
                            classical_bit_index,
                            inserted_operations,
                        )

                subcircuit = _finalize_subcircuit(subcircuit, qpd_qubits)
                cur_set_circuits[id_meas_subcircuit_index] = subcircuit
            obs_set_circuits.append(cur_set_circuits)
        experiment_circuits.append(obs_set_circuits)
    return CutExperiment(
        experiment_circuits,
        cut_circuit.cut_locations,
        cut_circuit.map_qubit,
        coefficients,
        id_meas[:num_id_meas],
        observables,
        backend=backend,
    )


def run_experiments(
    experiment_circuits: CutCircuit,
    cut_locations: np.ndarray[CutLocation],
    id_meas: list[tuple[int, int, int]],
    shots: int = 2**12,
    backend: None = None,
) -> list[TotalResult]:
    """Run experiment circuits.

    Loop through experiment circuits and then loop through circuit group and run each
    circuit. Store results as [group0, group1, ...] where group is [res0, res1, ...].
    where res is "xxx yy": count xxx are the measurements from the end of circuit
    measurements on the meas classical register and yy are the qpd basis measurement
    results from the qpd_meas class register.

    Args:
        experiment_circuits (CutCircuit): experiment circuits
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        id_meas (list[int, int, int]): list of identity basis measurement locations
        shots (int): number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)

    Returns:
        list[TotalResult]:
            list of transformed results

    """
    wire_cuts = len([i for i in cut_locations if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = len(cut_locations) - wire_cuts
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    samples = int(samples / experiment_circuits.num_groups)
    if backend is None:
        backend = AerSimulator()

    results = [0] * (experiment_circuits.num_groups)

    for count, subcircuit_group in enumerate(experiment_circuits.circuits):
        sub_result = [
            {
                " " + k: v
                for k, v in backend.run(i, shots=shots).result().get_counts().items()
            }
            if len(i.cregs) == 1 and i.cregs[0].name == "qpd_meas"
            else {" ": shots}
            if len(i.data) == 0 or i.data[-1].operation.name != "measure"
            else backend.run(i, shots=shots).result().get_counts()
            for i in subcircuit_group
        ]

        results[count] = sub_result

        sub_result = []
    return _process_results(results, id_meas, shots, samples)

def run_experiments_old(
    experiment_circuits: CutCircuit,
    cut_locations: np.ndarray[CutLocation],
    id_meas: list[tuple[int, int, int]],
    shots: int = 2**12,
    backend: None = None,
) -> list[TotalResult]:
    """Run experiment circuits.

    Loop through experiment circuits and then loop through circuit group and run each
    circuit. Store results as [group0, group1, ...] where group is [res0, res1, ...].
    where res is "xxx yy": count xxx are the measurements from the end of circuit
    measurements on the meas classical register and yy are the qpd basis measurement
    results from the qpd_meas class register.

    Args:
        experiment_circuits (CutCircuit): experiment circuits
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        id_meas (list[int, int, int]): list of identity basis measurement locations
        shots (int): number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)

    Returns:
        list[TotalResult]:
            list of transformed results

    """
    wire_cuts = len([i for i in cut_locations if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = len(cut_locations) - wire_cuts
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    samples = int(samples / experiment_circuits.num_groups)
    if backend is None:
        backend = AerSimulator()

    results = [0] * (experiment_circuits.num_groups)

    for count, subcircuit_group in enumerate(experiment_circuits.circuits):
        sub_result = [
            {
                " " + k: v
                for k, v in backend.run(i, shots=shots).result().get_counts().items()
            }
            if len(i.cregs) == 1 and i.cregs[0].name == "qpd_meas"
            else {" ": shots}
            if len(i.data) == 0 or i.data[-1].operation.name != "measure"
            else backend.run(i, shots=shots).result().get_counts()
            for i in subcircuit_group
        ]

        results[count] = sub_result

        sub_result = []
    return _process_results(results, id_meas, shots, samples)

def run_experiments(
    cut_experiment: CutExperiment,
    shots: int = 2**12,
    backend: None = None,
) -> list[list[TotalResult]]:
    """Run experiment circuits.

    Loop through experiment circuits and then loop through circuit group and run each
    circuit. Store results as [group0, group1, ...] where group is [res0, res1, ...].
    where res is "xxx yy": count xxx are the measurements from the end of circuit
    measurements on the meas classical register and yy are the qpd basis measurement
    results from the qpd_meas class register.

    Args:
        experiment_circuits (CutCircuit): experiment circuits
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        id_meas (list[int, int, int]): list of identity basis measurement locations
        shots (int): number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)

    Returns:
        list[TotalResult]:
            list of transformed results

    """
    wire_cuts = len([i for i in cut_experiment.cut_locations if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = len(cut_experiment.cut_locations) - wire_cuts
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    samples = int(samples / cut_experiment.num_groups)
    if backend is None:
        backend = AerSimulator()

    results: list[list[int, dict[str, int]]] = [0] * (cut_experiment.num_obs_groups)

    for count, circuit_group in enumerate(cut_experiment.experiments):
        group = []
        for obs_ind, obs_group in enumerate(circuit_group):
            obs_res = {}
            for ind, subcircuit in obs_group.items():
                sub_result = (
                    {" " + k: v
                     for k, v in backend.run(subcircuit, shots=shots).result().get_counts().items()}
                )
                obs_res[ind] = sub_result
            group.append(obs_res)
        results[count] = group

    all_keys = results[0][0].keys()

    for ind, sub_result in enumerate(results):
        for exp_ind, experiment_run in enumerate(sub_result):
            if experiment_run.keys() != all_keys:
                for key, val in results[0][0].items():
                    if key not in experiment_run:
                        experiment_run[key] = val

    return _process_results(results, cut_experiment.id_meas, shots, samples)


def _process_results_old(
    results: list[list[dict[str,int]]],
    id_meas: list[tuple[int, int, int]],
    shots: int,
    samples: int,
) -> list[TotalResult]:
    """Transform results with post processing function {0,1} -> [-1, 1].

    Tranform results so that we map 0 -> -1 and 1 -> 1. Gives processed results in form
    [TotalResult0, TotalResult1, ...], where TotalResult is
    [SubResult0, SubResult1, ...] and SubResult are [[[x0,x0,x0], [y0,y0], counts0],
    [[x1,x1,x1], [y1,y1], counts1], ...].

    Args:
        results (list): results from experiment circuits
        id_meas (list): locations of identity basis measurements
        shots (int): number of shots per circuit run
        samples (int): number of needed samples

    Returns:
    -------
        list[TotalResult]:
            list of transformed results

    """
    preocessed_results = []
    for experiment_run in results:
        experiment_run_results = []
        for sub_result in experiment_run:
            circuit_results = []
            for meassurements, count in sub_result.items():
                # separate end measurements from mid-circuit measurements
                if meassurements == " ":
                    separate_measurements = [meassurements.split(" ")[0]]
                else:
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

def _process_results(
    results: list[list[dict[str,int]]],
    id_meas: list[tuple[int, int, int]],
    shots: int,
    samples: int,
) -> list[list[TotalResult]]:
    """Transform results with post processing function {0,1} -> [-1, 1].

    Tranform results so that we map 0 -> -1 and 1 -> 1. Gives processed results in form
    [TotalResult0, TotalResult1, ...], where TotalResult is
    [SubResult0, SubResult1, ...] and SubResult are [[[x0,x0,x0], [y0,y0], counts0],
    [[x1,x1,x1], [y1,y1], counts1], ...].

    Args:
        results (list): results from experiment circuits
        id_meas (list): locations of identity basis measurements
        shots (int): number of shots per circuit run
        samples (int): number of needed samples

    Returns:
    -------
        list[TotalResult]:
            list of transformed results

    """
    preocessed_results = []

    for group_ind, circ_group in enumerate(results):
        for exp_ind, experiment_run in enumerate(circ_group):
            experiment_run_results = [0] * len(experiment_run)
            for sub_ind, sub_result in experiment_run.items():
                circuit_results = []
                for meassurements, count in sub_result.items():
                    # separate end measurements from mid-circuit measurements
                    if meassurements == " ":
                        separate_measurements = [meassurements.split(" ")[0]]
                    else:
                        separate_measurements = meassurements.split(" ")

                    # map to eigenvalues
                    result_eigenvalues = [
                        np.array([-1 if x == "0" else 1 for x in i])
                        for i in separate_measurements
                    ]
                    circuit_results.append(
                        SubResult(result_eigenvalues, count / shots * samples)
                    )
                experiment_run_results[sub_ind] = circuit_results
            if group_ind >= len(preocessed_results):
                preocessed_results.append([])
            preocessed_results[group_ind].append(TotalResult(experiment_run_results))
        
    return preocessed_results


# Calculate the approx expectation values for the original circuit
def estimate_expectation_values_old(
    results: list[TotalResult],
    coefficients: list[int],
    cut_locations: np.ndarray[CutLocation],
    observables: list[int | list[int]],
    map_qubits: Optional[dict[int, int]] = None,
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
        results (list[TotalResult]): results from experiment circuits
        coefficients (list[int]): list of coefficients for each subcircuit group
        cut_locations (np.ndarray[CutLocation]): cut locations
        observables (list[int | list[int]]):
            observables to calculate expectation values for

    Returns:
        list[float]:
            expectation values as a list of floats

    """
    cuts = len(cut_locations)
    wire_cuts = len([i for i in cut_locations if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = cuts - wire_cuts
    # number of samples neede
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    shots = int(samples / len(results))

    sum_shots = 0
    # ininialize approx expectation values of an array of ones
    expectation_values = np.ones(len(observables))
    for experiment_run, coefficient in zip(results, coefficients):
        # add sub results to the total approx expectation value
        mid = (
            np.power(-1, wire_cuts + 1)  # * (np.power(-1, cz_cuts)
            * coefficient
            * _get_sub_expectation_values(
                experiment_run, observables, shots, map_qubits
            )
        )
        sum_shots += shots
        expectation_values += mid

    # multiply by gamma to the power of cuts and take mean
    return np.power(4, wire_cuts) * np.power(3, cz_cuts) * expectation_values / samples


def _get_sub_expectation_values_old(
    experiment_run: TotalResult,
    observables: list[int | list[int]],
    shots: int,
    map_qubits: Optional[dict[int, int]] = None,
) -> list:
    """Calculate sub expectation value for the result.

    Args:
        experiment_run (TotalResult): results of a subcircuit pair
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z-observables)
        shots (int): number of shots

    Returns:
        list:
            list of sub expectation values

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
        if map_qubits is not None:
            sorted_full_result = np.array(
                [
                    full_result[map_qubits[key]]
                    for key in sorted(map_qubits.keys(), reverse=True)
                ]
            )
        else:
            sorted_full_result = full_result
        qpd_measurement_coefficient = 1  # initial value for qpd
        weight = shots  # initial weight
        for res in circuit_result:  # calculate weight and qpd coefficient
            weight *= res.count / shots
            # if len(res.measurements) > 1:
            qpd_measurement_coefficient *= np.prod(res.measurements[1])
        observable_results = np.empty(len(observables))  # initialize empty array
        # for obsrvables
        for count, obs in enumerate(observables):  # populate observable array
            if isinstance(obs, int):
                observable_results[count] = sorted_full_result[obs]  # if single qubit
            # observable just save
            # to array
            else:  # if multi qubit observable
                multi_qubit_observable_eigenvalue = 1  # initial eigenvalue
                for sub_observables in obs:  # multio qubit observable
                    multi_qubit_observable_eigenvalue *= sorted_full_result[
                        sub_observables
                    ]
                    observable_results[count] = (
                        np.power(-1, len(obs) + 1) * multi_qubit_observable_eigenvalue
                    )

        observable_expectation_value = (
            qpd_measurement_coefficient * observable_results * weight
        )
        sub_expectation_value += observable_expectation_value

    return sub_expectation_value

def estimate_expectation_values(
    results: list[list[TotalResult]],
    expv_data: dict
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
        results (list[TotalResult]): results from experiment circuits
        coefficients (list[int]): list of coefficients for each subcircuit group
        cut_locations (np.ndarray[CutLocation]): cut locations
        observables (list[int | list[int]]):
            observables to calculate expectation values for

    Returns:
        list[float]:
            expectation values as a list of floats

    """
    cuts = len(expv_data["cut_locations"])
    wire_cuts = len([i for i in expv_data["cut_locations"] if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = cuts - wire_cuts
    # number of samples neede
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    shots = int(samples / len(results))

    measurement_settings = get_needed_measurements_per_qubit(combine_measurements(
        expv_data["observables"]
    ))

    sum_shots = 0
    # ininialize approx expectation values of an array of ones
    expectation_values = np.ones(len(expv_data["observables"]))
    for experiment_run, coefficient in zip(results, expv_data["coefficients"]):
        # add sub results to the total approx expectation value
        mid = (
            np.power(-1, wire_cuts + 1)  # * (np.power(-1, cz_cuts)
            * coefficient
            * _get_sub_expectation_values(
                experiment_run, expv_data["observables"], shots, expv_data["map_qubits"], measurement_settings
            )
        )
        sum_shots += shots
        expectation_values += mid

    # multiply by gamma to the power of cuts and take mean
    return np.power(4, wire_cuts) * np.power(3, cz_cuts) * expectation_values / samples


def _get_sub_expectation_values(
    experiment_run: TotalResult,
    observables: list[int | list[int]],
    shots: int,
    map_qubits: Optional[dict[int, int]],
    measurement_settings: dict[int, set[str]],
) -> list:
    """Calculate sub expectation value for the result.

    Args:
        experiment_run (TotalResult): results of a subcircuit pair
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z-observables)
        shots (int): number of shots

    Returns:
        list:
            list of sub expectation values

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
        if map_qubits is not None:
            sorted_full_result = np.array(
                [
                    full_result[map_qubits[key]]
                    for key in sorted(map_qubits.keys(), reverse=True)
                ]
            )
        else:
            sorted_full_result = full_result
        qpd_measurement_coefficient = 1  # initial value for qpd
        weight = shots  # initial weight
        for res in circuit_result:  # calculate weight and qpd coefficient
            weight *= res.count / shots
            # if len(res.measurements) > 1:
            qpd_measurement_coefficient *= np.prod(res.measurements[1])
        observable_results = np.empty(len(observables))  # initialize empty array
        # for obsrvables
        for count, obs in enumerate(observables):  # populate observable array
            if isinstance(obs, int):
                observable_results[count] = sorted_full_result[obs]  # if single qubit
            # observable just save
            # to array
            else:  # if multi qubit observable
                multi_qubit_observable_eigenvalue = 1  # initial eigenvalue
                for sub_observables in obs:  # multio qubit observable
                    multi_qubit_observable_eigenvalue *= sorted_full_result[
                        sub_observables
                    ]
                    observable_results[count] = (
                        np.power(-1, len(obs) + 1) * multi_qubit_observable_eigenvalue
                    )

        observable_expectation_value = (
            qpd_measurement_coefficient * observable_results * weight
        )
        sub_expectation_value += observable_expectation_value

    return sub_expectation_value

