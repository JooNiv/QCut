"""
Utility functions for running on real backends.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile

from QCut.cutcircuit import CutCircuit
from QCut.cutlocation import CutLocation, SingleQubitCutLocation


def transpile_subcircuits(subcircuits: list[QuantumCircuit], 
                          cut_locations: list,
                          backend,
                          optimization_level: int = 0,
                          transpile_options: dict = None) -> CutCircuit:
    """
    Transpile subcircuits for a given backend. More efficient than transpiling
    experiment circuits as it only transpiles each subcircuit once instead of
    each experiment circuit. However, may lead to suboptimal transpilation results as
    the tranpiler cannot use the backend object directly due to need to retain
    some placeholder gates for cuts and observables. `transpile_options` can be used
    to pass additional options to the transpiler. For more control over transpilation
    of experiment circuits, use `transpile_experiments` or manually transpile them.

    Args:
        subcircuits (list[QuantumCircuit]): List of subcircuits to be transpiled.
        backend: Backend to transpile to.
        optimization_level (int): Optimization level for transpilation (0-3).
        transpile_options (dict): Arguments passed to qiskit transpile function.
    Returns:
        CutCircuit: Transpiled subcircuits wrapped in CutCircuit class.
    """

    basis  = []
    
    placeholders = []

    for ind, cut in enumerate(cut_locations):
        if isinstance(cut, SingleQubitCutLocation):
            placeholders.append(f"Meas_{ind}")
            placeholders.append(f"Init_{ind}")
        elif isinstance(cut, CutLocation):
            placeholders.append(f"cutCZ_c_{ind}")
            placeholders.append(f"cutCZ_t_{ind}")

    for i in range(sum(subcircuits.num_qubits for subcircuits in subcircuits)):
        placeholders.append(f"obs_{i}")
    
    if transpile_options and "basis_gates" in transpile_options:
        basis = transpile_options["basis_gates"]
        transpile_options.pop("basis_gates")
    else:
        try:
            basis = backend.configuration().basis_gates
        except Exception:
            basis = list(backend.architecture.gates.keys())
            basis = ["r" if gate == "prx" else gate for gate in basis]
    
    if transpile_options and "backend" in transpile_options:
        transpile_options.pop("backend")

    transpiled = transpile(subcircuits,
                           coupling_map=backend._coupling_map,
                           basis_gates=basis + placeholders + ["id"],
                           optimization_level=optimization_level,
                           **(transpile_options or {}))

    return CutCircuit(subcircuits=transpiled, backend=backend)

def transpile_experiments(experiment_circuits: list | CutCircuit, 
                          backend,
                          transpile_options: dict = None) -> CutCircuit:
    """
    Transpile experiment circuits. Transpiles all generated experiment circuits for
    a given backend. Most often one should use `transpile_subcircuits` instead, as that
    only subcircuits before experiment generation which is alot more efficient. This 
    function is mainly provided for special cases where one needs/wants extra control
    over the tranpilation of experiment circuits.

    Args:
        experiment_circuits: (list): Experiment circuits to be transpiled.
        backend (str): Backend to transpile to.
        transpile_options (dict): Arguments passed to qiskit transpile function.

    Returns:
        CutCircuit: Transpiled experiment circuits wrapped in CutCircuit class.
    """

    if isinstance(experiment_circuits, CutCircuit):
        experiment_circuits = experiment_circuits.circuits

    subexperiments = [
        [
            transpile(circuit, backend=backend, **(transpile_options or {}))
            for circuit in circuit_group
        ]
        for circuit_group in experiment_circuits
    ]

    return CutCircuit(subexperiments)


def run_and_expectation_value(
    circuit: QuantumCircuit, backend, observables: list, shots: int
) -> tuple[dict, list]:
    """Run circuit and calculate expectation value.

    Args:
        circuit (QuantumCircuit): A quantum circuit.
        backend: Backend to run circuit on.
        observables (list): Observables to calculate expectation values for.
        shots (int): Number of shots.

    Returns:
        tuple: A tuple containing:
            - dict: Counts from the circuit run.
            - list: A list of expectation values.
    """
    counts = run_on_backend(circuit, backend, shots)
    
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
