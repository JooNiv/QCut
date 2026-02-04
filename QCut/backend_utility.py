"""
Utility functions for running on real backends.
"""

from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import Gate
from qiskit.transpiler import Target

from QCut.cutcircuit import CutCircuit
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.wirecut import _remove_idle_wires


def transpile_subcircuits(cut_circuit: CutCircuit,
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

    custom_gates = {}
    for ind, i in enumerate(cut_circuit.cut_locations):
        if isinstance(i, CutLocation):
            custom_gates[f"cutCZ_t_{ind}"] = Gate(
                num_qubits=1, name=f"cutCZ_t_{ind}", params=[], label=f"cutCZ_t_{ind}"
            )
            custom_gates[f"cutCZ_c_{ind}"] = Gate(
                num_qubits=1, name=f"cutCZ_c_{ind}", params=[], label=f"cutCZ_c_{ind}"
            )

        elif isinstance(i, SingleQubitCutLocation):
            custom_gates[f"Meas_{ind}"] = Gate(
                num_qubits=1, name=f"Meas_{ind}", params=[], label=f"Meas_{ind}"
            )
            custom_gates[f"Init_{ind}"] = Gate(
                num_qubits=1, name=f"Init_{ind}", params=[], label=f"Init_{ind}"
            )
    
    for i in range(sum([x.num_qubits for x in cut_circuit.subcircuits])):
        custom_gates[f"obs_{i}"] = Gate(
            num_qubits=1, name=f"obs_{i}", params=[], label=f"obs_{i}"
        )

    target = Target()

    try:
        basis_gates = list({i[0].name for i in backend._target.instructions})
    except Exception:
        return cut_circuit
    
    target = target.from_configuration(
            num_qubits=backend.num_qubits,
            coupling_map=backend._coupling_map,
            basis_gates=basis_gates + list(custom_gates.keys()),
            custom_name_mapping=custom_gates,
        )


    transpiled = transpile(cut_circuit.subcircuits,
                           target=target,
                           optimization_level=optimization_level,
                           **(transpile_options or {}))

    transpiled = [_remove_idle_wires(circ) for circ in transpiled]

    return CutCircuit(subcircuits=transpiled, cut_locations=cut_circuit.cut_locations, 
                      map_qubit=cut_circuit.map_qubit, backend=backend)

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
