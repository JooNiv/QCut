"""Circuit knitting wire cut functionality."""

from __future__ import annotations

import pickle
from itertools import product
from typing import TYPE_CHECKING, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit_aer import AerSimulator
from qiskit_experiments.library import LocalReadoutError

from QCut.cutcircuit import CutCircuit
from QCut.cutlocation import CutLocation
from QCut.identity_qpd import identity_qpd
from QCut.qcutresult import SubResult, TotalResult

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
    for circ in subcircuits:
        subops = []
        for ind, op in enumerate(circ):
            if "Meas" in op.operation.name or "Init" in op.operation.name:
                subops.append((ind, op))
        ops.append(subops)

    return ops


def _remove_obsm(subcircuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
    for i in subcircuits:
        j = 0
        while j < len(i.data):
            if "obs" in i[j].operation.name:
                i.data.remove(i[j])
            else:
                j += 1


def get_experiment_circuits(  # noqa: C901
    subcircuits: list[QuantumCircuit],  # noqa: C901
    cut_locations: np.ndarray[CutLocation],
) -> tuple[CutCircuit, list[int], list[tuple[int, int, int]]]:
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

    _remove_obsm(subcircuits)

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
            subcircuit = pickle.loads(pickle.dumps(circ))
            # subcircuit = deepcopy(circ)
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
    return CutCircuit(experiment_circuits), coefficients, id_meas[:num_id_meas]


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
    experiment_circuits: CutCircuit,
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
        experiment_circuits (CutCircuit): experiment circuits
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        id_meas (list[int, int, int]): list of identity basis measurement locations
        shots (int): number of shots per circuit run (optional)
        backend: backend used for running the circuits (optional)
        mitigate (bool): wether to use readout error mitigation or not (optional)

    Returns:
        list[TotalResult]:
            list of transformed results

    """
    cuts = len(cut_locations)
    # number of samples neede
    samples = int(np.power(4, (2) * cuts) / np.power(ERROR, 2))
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
            if i.data[-1].operation.name != "measure"
            else backend.run(i, shots=shots).result().get_counts()
            for i in subcircuit_group
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
            * _get_sub_expectation_values(
                experiment_run, observables, shots, map_qubits
            )
        )
        expectation_values += mid

    # multiply by gamma to the power of cuts and take mean
    return np.power(4, cuts) * expectation_values / (samples)


def _get_sub_expectation_values(
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
