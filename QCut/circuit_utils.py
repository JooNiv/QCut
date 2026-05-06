"""
Utility functions for working with quantum circuits in the context of circuit
cutting and knitting.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def _count_gates(circuit: QuantumCircuit) -> dict[Qubit, int]:
    """Count the number of gates acting on each qubit in a QuantumCircuit.

    Args:
        circuit (QuantumCircuit): The input quantum circuit.

    Returns:
        dict[Qubit, int]: A dictionary mapping each qubit to the number of gates
        acting on it.
    """
    gate_count = dict.fromkeys(circuit.qubits, 0)
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            gate_count[qubit] += 1

    return gate_count


def _remove_idle_wires(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove idle wires from a QuantumCircuit.

    Args:
        circuit (QuantumCircuit): The input quantum circuit.

    Returns:
        QuantumCircuit: A new quantum circuit with idle wires removed.
    """
    gate_count = _count_gates(circuit)
    for qubit, count in gate_count.items():
        if count == 0:
            circuit.qubits.remove(qubit)

    return circuit


def _remove_obsm(subcircuits: list[dict[int, QuantumCircuit]]):

    for obs_set in subcircuits:
        for ind, circ in obs_set.items():
            j = 0
            while j < len(circ.data):
                if "obs" in circ[j].operation.name:
                    circ.data.remove(circ[j])
                else:
                    j += 1


def _remove_obsm_2(subcircuits: list[QuantumCircuit]):

    for circ in subcircuits:
        j = 0
        while j < len(circ.data):
            if "obs" in circ[j].operation.name:
                circ.data.remove(circ[j])
            else:
                j += 1
