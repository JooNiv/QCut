"""
Utility functions for running on real backends.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_experiments.library import LocalReadoutError


def transpile_experiments(experiment_circuits: list, backend) -> list:
    """
    Transpile experiment circuits.

    Args:
        experiment_circuits: (list): Experiment circuits to be transpiled.
        backend (str): Backend to transpile to.

    Returns:
        list: A list of transpiled experiment circuits.
    """

    return [
        [
            transpile(circuit, backend, layout_method="sabre", optimization_level=3)
            for circuit in circuit_group
        ]
        for circuit_group in experiment_circuits
    ]


def run_and_expectation_value(
    circuit: QuantumCircuit, backend, observables: list, shots: int, mitigate=False
) -> tuple[dict, list]:
    """Run circuit and calculate expectation value.

    Args:
        circuit (QuantumCircuit): A quantum circuit.
        backend: Backend to run circuit on.
        observables (list): Observables to calculate expectation values for.
        shots (int): Number of shots.
        mitigate (bool): If True, use readout error mitigation.

    Returns:
        tuple: A tuple containing:
            - dict: Counts from the circuit run.
            - list: A list of expectation values.
    """
    counts = run_on_backend(circuit, backend, shots)
    if mitigate:
        q = list(counts.keys())
        qs = list(range(len(q[0])))
        exp = LocalReadoutError(qs)
        exp.analysis.set_options(verbose=False)
        result = exp.run(backend)
        mitigator = result.analysis_results("Local Readout Mitigator").value
        mitigated_quasi_probs = mitigator.quasi_probabilities(counts)
        probs_test = {
            f"{int(old_key):0{len(qs)}b}"[::-1]: mitigated_quasi_probs[old_key] * shots
            if mitigated_quasi_probs[old_key] > 0
            else 0
            for old_key in mitigated_quasi_probs
        }
        counts = probs_test
    exps = expectation_values(counts, observables, shots)

    return counts, exps


def expectation_values(counts: dict, observables: list, shots: int) -> list:
    """Calculate expectation values.

    Args:
        counts (dict):
            Counts obtained from circuit run, where keys are measurement outcomes and
            values are the number of times each outcome was observed.

        observables (list):
            List of observables to calculate expectation values for. Each observable can
            be an integer (index of a single qubit) or a list of integers
            (indices of multiple qubits).

        shots (int): Number of shots (total number of measurements).

    Returns:
        list: A list of expectation values for each observable.

    """
    # Convert results to a list of dicts with measurement values and counts
    measurements = [
        {"meas": [1 if bit == "0" else -1 for bit in meas], "count": count}
        for meas, count in counts.items()
    ]

    # Initialize an array to store expectation values for each observable
    exps = np.zeros(len(observables))

    # Calculate expectation values
    for measurement in measurements:
        meas_values = measurement["meas"]
        count = measurement["count"]
        for idx, observable in enumerate(observables):
            if isinstance(observable, int):
                exps[idx] += meas_values[observable] * count
            else:
                exps[idx] += np.prod([meas_values[zi] for zi in observable]) * count

    return np.array(exps) / shots


def run_on_backend(circuit: QuantumCircuit, backend, shots: int) -> dict:
    """Run a quantum circuit on a specified backend.

    Args:
        circuit (QuantumCircuit): The quantum circuit to be executed.
        backend (Backend): The backend to use for executing the circuit.
        shots (int): The number of shots (repetitions) to run the circuit.

    Returns:
        dict: A dictionary of counts from the circuit run.
    """
    job = backend.run(circuit, shots=shots)
    result = job.result()
    return result.get_counts()
