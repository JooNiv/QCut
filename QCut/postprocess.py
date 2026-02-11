"""
A module for post-processing the results obtained from running the cut circuits and
calculating the estimated expectation values based on the results and the provided observables.
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np

from QCut.basis_transform import (
    _combine_pauli_ops,
    _get_observable_circuit_index
)
from QCut.cutlocation import SingleQubitCutLocation
from QCut.qcutresult import SubResult, TotalResult

ERROR = 0.0000001

def _process_results(
    results: list,
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
            experiment_run_results = []
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
                experiment_run_results.append(circuit_results)
            if group_ind >= len(preocessed_results):
                preocessed_results.append([])
            preocessed_results[group_ind].append(TotalResult(experiment_run_results))
        
    return preocessed_results

def _get_sub_expectation_values(
    experiment_run: TotalResult,
    observables: list,
    shots: int,
    map_qubits: Optional[dict[int, int]] = None,
) -> np.ndarray:
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
    sub_circuit_result_combinations = product(*experiment_run.subcircuits[0]) # type: ignore[no-matching-overload]

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
    by total cut cost and divide by number of samples.

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