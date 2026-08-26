"""
A module for post-processing the results obtained from running the cut circuits and
calculating the estimated expectation values based on the results and the
provided observables.
"""

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np

from QCut.cutlocation import SingleQubitCutLocation
from QCut.execution.basis_transform import (
    _combine_pauli_ops,
    _get_observable_circuit_index,
)
from QCut.execution.qcutresult import RawResult, SubResult, TotalResult


def _process_results(
    results: list,
    shots: int,
    qpd_bits: dict[tuple[int, int, int], tuple[int, int]] | None = None,
) -> list[list[TotalResult]]:
    """Transform results with post processing function {0,1} -> [-1, 1].

    Transform results so that we map 0 -> -1 and 1 -> 1. Gives processed results in form
    [TotalResult0, TotalResult1, ...], where TotalResult is
    [SubResult0, SubResult1, ...] and SubResult are [[[x0,x0,x0], [y0,y0], counts0],
    [[x1,x1,x1], [y1,y1], counts1], ...].

    Each outcome is weighted by how often it came up, so the weight carried through to
    the estimator is that outcome's probability.

    The qpd measurement bits are picked out by how many there are rather than by looking
    for their register in the reported counts. A backend is free to rename or merge
    classical registers -- IQM's transpiler does, when it routes through a resonator --
    and a key split on whitespace quietly returns the wrong field when it does. The
    widths come from the experiment, which knows them because it built the circuits.

    Bits that were dropped for going unwritten are put back as -1, which is what an
    unwritten bit reads as, so the sign is the one the full register would have given.

    Args:
        results (list): results from experiment circuits
        shots (int): number of shots the counts were taken at
        qpd_bits (dict): per circuit, how many qpd bits it wrote and how many were
            dropped. Without it the qpd register is taken to be the last whitespace
            separated field, as it was before the widths were recorded.

    Returns:
    -------
        list[TotalResult]:
            list of transformed results

    """
    processed_results = []

    for group_ind, circ_group in enumerate(results):
        for exp_ind, experiment_run in enumerate(circ_group):
            experiment_run_results = []
            for sub_ind, sub_result in experiment_run.items():
                circuit_results = []
                layout = (qpd_bits or {}).get((group_ind, exp_ind, sub_ind))
                for measurements, count in sub_result.items():
                    if layout is None:
                        # No widths recorded: fall back to splitting the key. A circuit
                        # with nothing to write to its qpd register does not carry one,
                        # so there may be a single field; pad to two either way, since
                        # everything downstream reads the observable bits as the first
                        # and the qpd bits as the second.
                        fields = [f for f in measurements.split(" ") if f]
                        while len(fields) < 2:
                            fields.append("")
                        result_eigenvalues = [
                            np.array([-1 if x == "0" else 1 for x in field])
                            for field in fields
                        ]
                    else:
                        written, dropped = layout
                        bits = measurements.replace(" ", "")
                        split = len(bits) - written
                        qpd = [-1] * dropped + [
                            -1 if x == "0" else 1 for x in bits[split:]
                        ]
                        result_eigenvalues = [
                            np.array([-1 if x == "0" else 1 for x in bits[:split]]),
                            np.array(qpd),
                        ]
                    circuit_results.append(SubResult(result_eigenvalues, count / shots))
                experiment_run_results.append(circuit_results)
            if group_ind >= len(processed_results):
                processed_results.append([])
            processed_results[group_ind].append(TotalResult(experiment_run_results))

    return processed_results


def _get_sub_expectation_values(
    experiment_run: TotalResult,
    observables: list,
    map_qubits: Optional[dict[int, int]] = None,
) -> np.ndarray:
    """Calculate sub expectation value for the result.

    One subcircuit group's contribution: over every combination of the subcircuits'
    end-of-circuit outcomes, the product of their probabilities times the observable's
    eigenvalue times the sign the mid-circuit measurements carry.

    Args:
        experiment_run (TotalResult): results of a subcircuit pair
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z-observables)

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
        # concat results to one array and reverse to account for qiskit qubit ordering
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
        probability = 1.0  # joint probability of this combination of outcomes
        for res in circuit_result:
            probability *= res.count  # already a probability, see _process_results
            qpd_measurement_coefficient *= np.prod(res.measurements[1])
        observable_results = np.empty(len(observables))  # initialize empty array
        # for observables
        for count, obs in enumerate(observables):  # populate observable array
            if isinstance(obs, int):
                observable_results[count] = sorted_full_result[obs]  # if single qubit
            # observable just save
            # to array
            else:  # if multi qubit observable
                multi_qubit_observable_eigenvalue = 1  # initial eigenvalue
                for sub_observables in obs:  # multi qubit observable
                    multi_qubit_observable_eigenvalue *= sorted_full_result[
                        sub_observables
                    ]
                    observable_results[count] = (
                        np.power(-1, len(obs) + 1) * multi_qubit_observable_eigenvalue
                    )

        observable_expectation_value = (
            qpd_measurement_coefficient * observable_results * probability
        )
        sub_expectation_value += observable_expectation_value

    return sub_expectation_value


def estimate_expectation_values(
    results: RawResult, expv_data: dict | None = None
) -> np.ndarray:
    r"""Calculate the estimated expectation values.

    Loop through processed results. For each result group generate all products of
    different measurements from different subcircuits of the group. For each result
    from qpd measurements calculate qpd coefficient and from counts calculate weight.
    The estimate is the quasiprobability sum itself,

    .. math::

        \langle O \rangle = (-1)^{w+1} \sum_g c_g E_g

    over the subcircuit groups, where :math:`c_g` is the group's coefficient,
    :math:`E_g` is what :func:`_get_sub_expectation_values` returns for it, and
    :math:`w` counts the wire cuts. That parity is the qpd register's sign convention:
    every one of its bits maps 0 to -1, so an unwritten bit contributes -1 and only the
    number allocated per subcircuit survives.

    Multi-qubit observables pick up a further :math:`(-1)^{m+1}` for their :math:`m`
    qubits, applied while their eigenvalues are multiplied together.

    Args:
        results (RawResult): raw results from experiment circuits. Carries the data
            needed to interpret itself, so ``expv_data`` does not have to be passed.
        expv_data (dict, optional): experiment data, if it is not the data the results
            were produced with. Defaults to what ``results`` recorded.

    Returns:
        np.ndarray:
            one expectation value per observable, in the order they were given

    """
    raw_results = results
    if expv_data is None:
        expv_data = raw_results.expv_data
        if expv_data is None:
            raise ValueError(
                "These results carry no experiment data, so expv_data has to be given. "
                "Results from QCut.run_experiments carry it already."
            )
    results_processed = _process_results(
        raw_results.results, raw_results._shots, expv_data.get("qpd_bits")
    )

    wire_cuts = len(
        [i for i in expv_data["cut_locations"] if isinstance(i, SingleQubitCutLocation)]
    )
    parity = np.power(-1, wire_cuts + 1)

    measurement_settings = _combine_pauli_ops(expv_data["observables"])

    result_for_obs = []

    for obs in expv_data["observables"].paulis:
        obs_circuit_info = _get_observable_circuit_index(obs, measurement_settings)
        result_for_obs.append(obs_circuit_info)

    expectation_values = np.zeros(len(expv_data["observables"]))

    for ind, obs_data in enumerate(result_for_obs):
        if obs_data["circuit_index"] is None:
            raise ValueError("""Observable cannot be measured 
                             with given measurement settings.""")

        for experiment_run, coefficient in zip(
            results_processed, expv_data["coefficients"]
        ):
            cur_obs = (
                obs_data["obs_indices"]
                if len(obs_data["obs_indices"]) == 1
                else [obs_data["obs_indices"]]
            )
            expectation_values[ind] += (
                parity
                * coefficient
                * _get_sub_expectation_values(
                    experiment_run[obs_data["circuit_index"]],
                    cur_obs,
                    expv_data["map_qubit"],
                )
            )[0]

    return expectation_values
