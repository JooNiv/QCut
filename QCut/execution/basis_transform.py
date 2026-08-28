"""
Module for transforming measurement bases in quantum circuits
when generating experiment circuits, particularly for combining Pauli
operators into compatible measurement settings.
"""

from __future__ import annotations

from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit.circuit import (
    Instruction,
    QuantumRegister,
)
from qiskit.circuit.library import HGate, SdgGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass


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


class _ModifyMeasurementBasis(TransformationPass):
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
                    # continue
                    dag.remove_op_node(node)
                    break

                ob = setting[obs_ind]
                no_obs = False

                mini_dag = DAGCircuit()
                register = QuantumRegister(1)
                mini_dag.add_qreg(register)

                if ob == "X":
                    if self.ops and "X-meas" in self.ops:
                        mini_dag.apply_operation_back(self.ops["X-meas"], [register[0]])
                    else:
                        mini_dag.apply_operation_back(HGate(), [register[0]])
                elif ob == "Y":
                    if self.ops and "Y-meas" in self.ops:
                        mini_dag.apply_operation_back(self.ops["Y-meas"], [register[0]])
                    else:
                        mini_dag.apply_operation_back(SdgGate(), [register[0]])
                        mini_dag.apply_operation_back(HGate(), [register[0]])

                dag.substitute_node_with_dag(node, mini_dag)

        if no_obs:
            return cloned_dag
        return dag


def _get_obs_subcircuits(
    subcircuits: list[QuantumCircuit],
    measurement_settings: list[dict[int, str]],
    ops: dict[str, Instruction] | None = None,
) -> list[dict[int, QuantumCircuit]]:
    pms = [
        PassManager([_ModifyMeasurementBasis([setting], ops)])
        for setting in measurement_settings
    ]
    obs_subcircuits = []
    for pm in pms:
        pm_circs = {}
        for ind, subcircuit in enumerate(subcircuits):
            modified_circuit = pm.run(subcircuit)
            if modified_circuit.num_qubits == 0:
                continue

            modified_circuit._layout = subcircuit.layout
            pm_circs[ind] = modified_circuit
        obs_subcircuits.append(pm_circs)
    return obs_subcircuits


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
