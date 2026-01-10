"""Circuit knitting wire cut functionality."""

from __future__ import annotations

import pickle
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    CircuitInstruction,
    Instruction,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import HGate, SdgGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_aer import AerSimulator

from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.qcutresult import SubResult, TotalResult
from QCut.qpd import cz_qpd, identity_qpd

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


def _get_placeholder_locations(subcircuits: list[QuantumCircuit]) -> list:
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


def _remove_obsm(subcircuits: list[dict[int, QuantumCircuit]]
                 ) -> list[dict[int, QuantumCircuit]]:

    for obs_set in subcircuits:
        for ind, circ in obs_set.items():
            j = 0
            while j < len(circ.data):
                if "obs" in circ[j].operation.name:
                    circ.data.remove(circ[j])
                else:
                    j += 1

def _insert_wire_cut_qpd(
    ind,
    op,
    subcircuit,
    offset,
    qpd_qubits,
    qpd,
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
            # remove extra classical bits and registers
            #_adjust_cregs(subcircuit)
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

    return offset, classical_bit_index, inserted_operations


def _insert_cz_cut_qpd(  # noqa: C901
    ind,
    op,
    subcircuit,
    offset,
    qpd_qubits,
    qpd,
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
            # remove extra classical bits and registers
            #if meas_op.name != "id-meas":
            #    _adjust_cregs(subcircuit)
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
            
            # remove extra classical bits and registers
            #if meas_op.name != "id-meas":
            #    _adjust_cregs(subcircuit)
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

        inserted_operations += 1
        offset += len(meas_op) - 1

    return offset, classical_bit_index, inserted_operations

def _combine_pauli_ops(op: SparsePauliOp) -> list[dict[int, str]]:  # noqa: C901
    """Combine Pauli operators that have no conflicting non-identity components.
    
    Args:
        op (SparsePauliOp): The SparsePauliOp to analyze.
    
    Returns:
        list[dict[int, str]]: A list of combined measurement settings, where each dict
                              maps qubit indices to Pauli basis measurements.
    """
    pauli_strings = [pauli.to_label()[::-1] for pauli in op.paulis]
    
    combined_settings = []
    used = [False] * len(pauli_strings)
    
    for i, pauli_string in enumerate(pauli_strings):
        if used[i]:
            continue
        
        # Start a new combined setting with the current Pauli string
        combined = {}
        for qubit_index, pauli in enumerate(pauli_string):
            if pauli != "I":
                combined[qubit_index] = pauli
        
        used[i] = True
        
        # Try to combine with remaining Pauli strings
        for j in range(i + 1, len(pauli_strings)):
            if used[j]:
                continue
            
            # Check if pauli_strings[j] can be combined with current combined setting
            can_combine = True
            for qubit_index, pauli in enumerate(pauli_strings[j]):
                if pauli != "I":
                    if qubit_index in combined and combined[qubit_index] != pauli:
                        can_combine = False
                        break
            
            # If compatible, add to combined setting
            if can_combine:
                for qubit_index, pauli in enumerate(pauli_strings[j]):
                    if pauli != "I":
                        combined[qubit_index] = pauli
                used[j] = True
        
        combined_settings.append(combined)
    
    return combined_settings

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
        
        no_obs = True
        cloned_dag = deepcopy(dag)
        for node in dag.op_nodes():
            if "obs" not in node.op.name:
                continue
 
            obs_ind = int(node.op.name.split("_")[-1])
            
            for setting in self.measurement_settings:
                if obs_ind not in setting:
                    #continue
                    dag.remove_op_node(node)
                    break
                
                ob = setting[obs_ind]
                no_obs = False
                
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
            return cloned_dag
        return dag

def _get_obs_subcircuits(subcircuits: list[QuantumCircuit], 
                        measurement_settings: list[dict[int, str]],
                        ops: dict[str, Instruction] | None = None
                        ) -> list[dict[int, QuantumCircuit]]:
    pms = [PassManager([ModifyMeasurementBasis([setting], ops)]) 
           for setting in measurement_settings]
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

def _remove_obsm_2(subcircuits: list[dict[int, QuantumCircuit]]
                 ) -> list[dict[int, QuantumCircuit]]:

    for circ in subcircuits:
        j = 0
        while j < len(circ.data):
            if "obs" in circ[j].operation.name:
                circ.data.remove(circ[j])
            else:
                j += 1


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

    num_qubits = 0
    for subcircuit in cut_circuit.subcircuits:
        crs = subcircuit.cregs
        for cr in crs:
            if cr.name == "meas":
                num_qubits += cr.size

    if all(len(obs) != num_qubits for obs in observables.paulis):
        raise ValueError(
            f"""ALL observable lengths must match 
            the number of qubits in the original uncut circuit 
            ({num_qubits})."""
        )

    qpd_combinations = get_qpd_combinations(cut_circuit.cut_locations)  
    # generate the QPD
    # operation combinations

    check_circuit_type = cut_circuit.backend is not None
    
    measurement_settings = _combine_pauli_ops(observables)

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

        obs_subcircuits = _get_obs_subcircuits(
            cut_circuit.subcircuits, measurement_settings, ops
        )
    else:
        obs_subcircuits = _get_obs_subcircuits(
            cut_circuit.subcircuits, measurement_settings
        )



    _remove_obsm(obs_subcircuits)

    _remove_obsm_2(cut_circuit.subcircuits)

    # initialize solution lists
    cuts = len(cut_circuit.cut_locations)
    cz_cuts = len([i for i in cut_circuit.cut_locations if isinstance(i, CutLocation)])
    wire_cuts = cuts - cz_cuts

    num_circs = np.power(8, wire_cuts) * np.power(6, cz_cuts)
    experiment_circuits = []
    coefficients = np.empty(num_circs)
    placeholder_locations = _get_placeholder_locations(cut_circuit.subcircuits)
    for id_meas_experiment_index, qpd in enumerate(
        qpd_combinations
    ):  # loop through all
        # QPD combinations
        coefficients[id_meas_experiment_index] = np.prod([op["c"] for op in qpd])

        if check_circuit_type:
            for sub in qpd:
                sub["op_0"] = transpile(sub["op_0"], basis_gates=basis)
                sub["op_1"] = transpile(sub["op_1"], basis_gates=basis)

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
                qpd_qubits = []  # store the qubit indices of qubits used for qpd
                # measurements
                for op_ind in placeholder_locations[id_meas_subcircuit_index]:
                    ind, op = op_ind

                    actual_op = subcircuit.data[ind + offset]
                    
                    if actual_op.operation.name != op.operation.name:
                        cur_ind = ind + offset
                        for i in range(len(subcircuit.data)):
                            cur_ind_minus = cur_ind - i
                            cur_ind_plus = cur_ind + i
                            if op.operation.name == subcircuit.data[
                                cur_ind_minus].operation.name:
                                ind = cur_ind_minus - offset
                                break
                            if op.operation.name == subcircuit.data[
                                cur_ind_plus].operation.name:
                                ind = cur_ind_plus - offset
                                break
                            

                    if "cut" in op.operation.name:
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_cz_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            classical_bit_index,
                            inserted_operations,
                        )

                    elif "Meas" in op.operation.name or "Init" in op.operation.name:
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_wire_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            classical_bit_index,
                            inserted_operations,
                        )
                    else:
                        raise ValueError(
                            f"""Unknown placeholder operation: {op.operation.name}.
                            Actual operation: {subcircuit.data[ind + offset]}"""
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
        observables,
        backend=backend,
    )

def run_experiments(  # noqa: C901
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
        shots (int): number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)

    Returns:
        list[TotalResult]:
            list of transformed results

    """
    wire_cuts = len([i for i in cut_experiment.cut_locations 
                     if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = len(cut_experiment.cut_locations) - wire_cuts
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    samples = int(samples / cut_experiment.num_groups)
    if backend is None:
        backend = AerSimulator()

    results: list[list[int, dict[str, int]]] = [0] * (cut_experiment.num_groups)

    for count, circuit_group in enumerate(cut_experiment.experiments):
        group = []
        for obs_ind, obs_group in enumerate(circuit_group):
            obs_res = {}
            for ind, subcircuit in obs_group.items():
                try:
                    sub_result = dict(backend.run(
                            subcircuit, shots=shots
                        ).result().get_counts().items())
                except Exception:
                    sub_result = {" " + "0"*subcircuit.num_clbits: shots}
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
    
    return _process_results(results, shots, samples)

def _process_results(
    results: list[list[dict[str,int]]],
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

def _get_sub_expectation_values(
    experiment_run: TotalResult,
    observables: SparsePauliOp,
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

    for ind, circuit_result in enumerate(sub_circuit_result_combinations):  
        # loop through results
        # concat results to one array and reverse to account for qiskit quibit ordering
        full_result = np.concatenate(
            [i.measurements[0] for i in reversed(circuit_result)]
        )

        if full_result.size == 0:
            raise ValueError("No measurement results found. This should not happen.")
            continue
        
        if map_qubits is not None:
            sorted_full_result = np.array(
                [
                    full_result[map_qubits[key]]
                    for key in sorted(map_qubits.keys(), reverse=True)
                ]
            )
        else:
            sorted_full_result = full_result
        
        sorted_full_result = list(reversed(sorted_full_result))

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

def _get_observable_circuit_index(pauli, combined: list[dict[int, str]]):
    """Find which measurement setting covers the non-identity letters of `pauli`,
    and return the indices of the qubits involved."""
    label = pauli
    non_identity = {i: p for i, p in enumerate(label) if p.to_label() != "I"}

    for idx, setting in enumerate(combined):
        # All non-identity qubits must be measured in the matching basis
        if all(setting.get(q) == p.to_label() for q, p in non_identity.items()):
            return {"circuit_index": idx, "obs_indices": list(non_identity.keys())}

    return {"circuit_index": None, "obs_indices": []}

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
    wire_cuts = len([i for i in expv_data["cut_locations"] 
                      if isinstance(i, SingleQubitCutLocation)])
    cz_cuts = cuts - wire_cuts
    # number of samples neede
    samples = int(
        (np.power(4, 2 * wire_cuts) * np.power(3, 2 * cz_cuts)) / np.power(ERROR, 2)
    )
    shots = int(samples / len(results))

    measurement_settings = _combine_pauli_ops(expv_data["observables"])

    result_for_obs = []

    for obs in expv_data["observables"].paulis:
        obs_circuit_info = _get_observable_circuit_index(obs, measurement_settings)
        result_for_obs.append(obs_circuit_info)

    sum_shots = 0
    # ininialize approx expectation values of an array of ones
    expectation_values = np.ones(len(expv_data["observables"]))

    for ind, obs_data in enumerate(result_for_obs):
        if obs_data["circuit_index"] is None:
            raise ValueError("""Observable cannot be measured 
                             with given measurement settings.""")
        
        for experiment_run, coefficient in zip(results, expv_data["coefficients"]):
        # add sub results to the total approx expectation value
            cur_obs = (obs_data["obs_indices"] 
                       if len(obs_data["obs_indices"]) == 1 
                       else [obs_data["obs_indices"]])
            mid = (
                np.power(-1, wire_cuts + 1)  # * (np.power(-1, cz_cuts)
                * coefficient
                * _get_sub_expectation_values(
                    experiment_run[obs_data["circuit_index"]], cur_obs,
                    shots, expv_data["map_qubit"])
            )[0]
            sum_shots += shots
            expectation_values[ind] += mid

    # multiply by gamma to the power of cuts and take mean
    return np.power(4, wire_cuts) * np.power(3, cz_cuts) * expectation_values / samples