"""
A module for the main circuit knitting workflow.
"""

from __future__ import annotations

import pickle

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    Qubit,
)
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from QCut.backend_utility import transpile_subcircuits
from QCut.basis_transform import (
    _combine_pauli_ops,
    _get_obs_subcircuits,
)
from QCut.circuit_preparation import get_locations_and_subcircuits
from QCut.circuit_utils import _remove_obsm, _remove_obsm_2
from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.postprocess import ERROR, _process_results, estimate_expectation_values
from QCut.qcutresult import TotalResult
from QCut.qpd_operations import (
    _insert_cz_cut_qpd,
    _insert_wire_cut_qpd,
    get_qpd_combinations,
)


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
        for ind, op in enumerate(circ.data):
            # if "Meas" in op.operation.name or "Init" in op.operation.name :
            if any(i in op.operation.name for i in names):
                subops.append((ind, op))
        ops.append(subops)

    return ops

def get_experiment_circuits(  # noqa: C901
    cut_circuit: CutCircuit,
    observables: SparsePauliOp,
) -> CutExperiment:
    """Generate experiment circuits by inserting QPD operations on
    measure/initialize/cutCZ nodes.

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
            basis = backend.configuration().basis_gates # type: ignore[possibly-missing-attribute]
        except Exception:
            basis = list(backend.architecture.gates.keys()) # type: ignore[possibly-missing-attribute]
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
        coefficients, # type: ignore[invalid-argument-type]
        observables,
        backend=backend,
    )

def run_experiments(  # noqa: C901
    cut_experiment: CutExperiment,
    shots: int = 2**12,
    backend = None,
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

    results = [0] * (cut_experiment.num_groups)

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



def run_cut_circuit(
    cut_circuit: CutCircuit,
    observables: SparsePauliOp,
    backend=AerSimulator(),
) -> list[float]:
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
        subcircuits (list[QuantumCircuit]):
            subcircuits containing the placeholder operations
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z observable)
        backend: backend to use for running experiment circuits (optional)

    Returns:
        list: a list of expectation values

    """

    if not isinstance(backend, AerSimulator):
        transpiled_subcircuits = transpile_subcircuits(cut_circuit
                                                       ,backend,
                                                       optimization_level=3)
    
        cut_experiment = get_experiment_circuits(transpiled_subcircuits, 
                                        observables)
    else:
        cut_experiment = get_experiment_circuits(cut_circuit, 
                                        observables)
        
    results = run_experiments(
        cut_experiment,
        backend=backend,
    )

    return estimate_expectation_values(
        results, cut_experiment.expv_data()
    )


def run(
    circuit: QuantumCircuit,
    observables: SparsePauliOp,
    backend=AerSimulator(),
) -> list[float]:
    """Run the whole circuit knitting sequence with one function call.

    Args:
        circuit (QuantumCircuit): circuit with cut experiments
        observables (list[int | list[int]]):
            list of observbles in the form of qubit indices (Z-obsevable).
        backend: backend to use for running experiment circuits (optional)

    Returns:
        list: a list of expectation values

    """
    # circuit = circuit.copy()
    cut_circuit = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(cut_circuit, observables, backend)