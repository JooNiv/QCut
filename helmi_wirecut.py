"""Utility functions for running on real backends."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_experiments.library import LocalReadoutError



def transpile_experiments(experiment_circuits: list, backend) -> list:
    """Transpile experiment circuits.

    Args:
    ----
        experiment_circuits: experiment circuits
        backend: backend to transpile to

    Returns:
    -------
        transpiled_experiments: a list of transpiled experiment circuits

    """
    transpiled_experiments = []
    for circuit_group in experiment_circuits:
        transpiled_group = []
        for circuit in circuit_group:
            transpiled_circuit = transpile(circuit, backend, layout_method="sabre", optimization_level=3)
            transpiled_group.append(transpiled_circuit)
        transpiled_experiments.append(transpiled_group)

    return transpiled_experiments

def run_and_expectation_value(circuit: QuantumCircuit, backend, observables: list, shots: int, mitigate = False) -> tuple[dict, list]:  # noqa: ANN001
    """Run circuit and calculate expectation value

    Args:
    ----
        circuit: a quantum circuit
        backend: backend to run circuit on
        observables: observables to calculate expectation values for
        shots: number of shots
        mitigate: if True use readout error mitigation

    Returns:
    -------
        expectation_values: a list of expectation values

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
        probs_test = {f"{int(old_key):0{len(qs)}b}"[::-1]:
                        mitigated_quasi_probs[old_key]*shots if mitigated_quasi_probs[old_key] > 0 else 0
                            for old_key in mitigated_quasi_probs}
        counts = probs_test
    exps = expectation_values(counts, observables, shots)

    return counts, exps

def expectation_values(counts: dict, observables: list, shots: int) -> list:  # noqa: FBT001, FBT002
    """Calculate expectration values.

    Args:
    ----
        counts: counts obtained from circuit run
        observables: observables to calculate expectation values for
        shost: number of shots
        probs

    Returns:
    -------
        cut_locations: a list of cut locations
        subcircuits: subcircuits with placeholder operations

    """
    new_res = []
    for meas, count in counts.items():
        res = {"meas": [1 if x == "0" else -1 for x in meas], "count":count}
        new_res.append(res)

    exps = np.zeros(len(observables))
    for  i in new_res:
        for sub, z in enumerate(observables):
            if isinstance(z, int):
                exps[sub] += i["meas"][z]*i["count"]
            else:
                exps[sub] += np.prod([i["meas"][zi] for zi in z])*i["count"]
    return np.array(exps) / shots

def run_on_backend(circuit: QuantumCircuit, backend, shots: int) -> dict:  # noqa: ANN001
    """Run circuit on backend.

    Args:
    ----
        circuit: a quantum circuit to be executed
        backend: backend to use for executing circuit
        shost: number of shots
        probs

    Returns:
    -------
        dict: a dictionary of counts from circuit run

    """
    job = backend.run(circuit, shots=shots)
    result = job.result()
    return result.get_counts()
