"""
Utility functions for running on real backends.
"""

from __future__ import annotations

from qiskit import transpile
from qiskit.circuit import Gate
from qiskit.transpiler import Target

from QCut.circuit_utils import (
    _drop_barriers,
    _fence_markers,
    _to_logical_order,
    barriers_to_markers,
    markers_to_barriers,
)
from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation, SingleQubitCutLocation


def _iqm_transpiler_for(backend):
    """Return IQM's transpiler if it is installed and ``backend`` is one of its own.

    The adapter comes with ``pip install "QCut[iqm]"``. Without it, or for any other
    backend, this returns None and the ordinary qiskit path is used. The backend check
    is what ``IQMBackendBase`` is imported for: a fake IQM device and a real one both
    derive from it, and nothing else does.
    """
    try:
        from iqm.qiskit_iqm import transpile_to_IQM
        from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
    except ImportError:
        return None
    return transpile_to_IQM if isinstance(backend, IQMBackendBase) else None


def transpile_subcircuits(
    cut_circuit: CutCircuit,
    backend,
    optimization_level: int = 0,
    transpile_options: dict | None = None,
    use_iqm_transpiler: bool = True,
) -> CutCircuit:
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

    if not isinstance(cut_circuit, CutCircuit):
        raise ValueError("cut_circuit must be of type CutCircuit.")

    custom_gates = {}
    for ind, i in enumerate(cut_circuit.cut_locations):
        if isinstance(i, CutLocation):
            custom_gates[f"cut{i.gate_name.upper()}_t_{ind}"] = Gate(
                num_qubits=1,
                name=f"cut{i.gate_name.upper()}_t_{ind}",
                params=[],
                label=f"cut{i.gate_name.upper()}_t_{ind}",
            )
            custom_gates[f"cut{i.gate_name.upper()}_c_{ind}"] = Gate(
                num_qubits=1,
                name=f"cut{i.gate_name.upper()}_c_{ind}",
                params=[],
                label=f"cut{i.gate_name.upper()}_c_{ind}",
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

    except Exception as e:
        raise ValueError(f"Error accessing backend target instructions: {e}")

    target = target.from_configuration(
        num_qubits=backend.num_qubits,
        coupling_map=backend._coupling_map,
        basis_gates=basis_gates + list(custom_gates.keys()),
        custom_name_mapping=custom_gates,
    )

    # Fenced first: a placeholder is a single-qubit gate as far as the transpiler is
    # concerned, so without barriers it gets commuted past the others and the experiment
    # builder, which reads them in order, misreads the result.
    marker_names = set(custom_gates)

    iqm_transpile = _iqm_transpiler_for(backend) if use_iqm_transpiler else None
    if iqm_transpile is not None:
        # IQM's own transpiler produces markedly shallower circuits than the generic
        # path, and it will accept the placeholders once they are barriers carrying
        # their name. Two of its defaults have to go the other way for QCut:
        #
        #   remove_final_rzs   a Z rotation before a Z measurement is unobservable, so
        #                      it drops trailing ones. QCut adds the rotations for X and
        #                      Y observables later, and those frames are needed then.
        #   perform_move_routing
        #                      rebuilds the classical registers on the way to the Star
        #                      architecture and loses ``qpd_meas`` wherever a term does
        #                      not write to it. Inserting the MOVEs belongs at
        #                      submission, which is where iqm-client does it.
        #
        # Both are still overridable through ``transpile_options``, deliberately, but
        # the answers will be wrong.
        options = {
            "remove_final_rzs": False,
            "perform_move_routing": False,
            "optimization_level": optimization_level,
        }
        options.update(transpile_options or {})
        transpiled = [
            barriers_to_markers(
                _to_logical_order(
                    iqm_transpile(
                        markers_to_barriers(subcircuit, marker_names),
                        backend,
                        **options,
                    ),
                    subcircuit.num_qubits,
                ),
                marker_names,
            )
            for subcircuit in cut_circuit.subcircuits
        ]
        return CutCircuit(
            subcircuits=transpiled,
            cut_locations=cut_circuit.cut_locations,
            map_qubit=cut_circuit.map_qubit,
            options=cut_circuit.options,
            backend=backend,
        )

    transpiled = transpile(
        [_fence_markers(circuit, marker_names) for circuit in cut_circuit.subcircuits],
        target=target,
        optimization_level=optimization_level,
        **(transpile_options or {}),
    )

    # Undo the layout: transpiling lays the subcircuits out on physical qubits and pads
    # them to the device width, and post-processing reads measurement bits by position.
    transpiled = [
        _drop_barriers(_to_logical_order(circuit, original.num_qubits))
        for circuit, original in zip(transpiled, cut_circuit.subcircuits)
    ]

    return CutCircuit(
        subcircuits=transpiled,
        cut_locations=cut_circuit.cut_locations,
        map_qubit=cut_circuit.map_qubit,
        options=cut_circuit.options,
        backend=backend,
    )


def transpile_experiments(
    cut_experiment: CutExperiment,
    backend,
    optimization_level: int = 0,
    transpile_options: dict | None = None,
) -> CutExperiment:
    """
    Transpile experiment circuits. Transpiles all generated experiment circuits for
    a given backend. Most often one should use `transpile_subcircuits` instead, as that
    only transpiles subcircuits before experiment generation which is a lot more
    efficient. This function is mainly provided for special cases where one needs/wants
    extra control over the transpilation of experiment circuits.

    Args:
        cut_experiment: (CutExperiment): Experiment circuits to be transpiled.
        backend (str): Backend to transpile to.
        optimization_level (int): Optimization level for transpilation (0-3).
        transpile_options (dict): Arguments passed to qiskit transpile function.

    Returns:
        CutExperiment: Transpiled experiment circuits wrapped in CutExperiment class.
    """

    if not isinstance(cut_experiment, CutExperiment):
        raise ValueError("cut_experiment must be of type CutExperiment.")

    subexperiments = [
        [
            {
                ind: transpile(
                    circ,
                    backend=backend,
                    optimization_level=optimization_level,
                    **(transpile_options or {}),
                )
                for ind, circ in exp.items()
            }
            for exp in exps
        ]
        for exps in cut_experiment.experiments
    ]

    return CutExperiment(
        subexperiments,
        cut_locations=cut_experiment.cut_locations,
        map_qubit=cut_experiment.map_qubit,
        coefficients=cut_experiment.coefficients,
        observables=cut_experiment.observables,
        options=cut_experiment.options,
        backend=backend,
    )
