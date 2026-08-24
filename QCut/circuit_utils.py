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


def _fence_markers(circuit: QuantumCircuit, names: set[str]) -> QuantumCircuit:
    """Put a barrier either side of every placeholder, so transpiling cannot move it.

    The experiment builder walks a subcircuit's instructions in order and expects the
    placeholders where it left them. Transpiling is free to commute single-qubit gates
    past each other, and a placeholder looks like one, so without a fence the markers
    come back permuted and the walk misreads the circuit.

    Fencing is also the more honest instruction to give the transpiler. A placeholder
    stands for an operation that has not been chosen yet, so merging the gates on either
    side of it across the gap is not a valid simplification in the first place.
    """
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        fenced = instruction.operation.name in names
        if fenced:
            out.barrier(instruction.qubits)
        out.append(instruction.operation, instruction.qubits, instruction.clbits)
        if fenced:
            out.barrier(instruction.qubits)
    return out


def _drop_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove the fences again, once transpilation can no longer reorder anything."""
    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name != "barrier":
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
    return out


def _to_logical_order(circuit: QuantumCircuit, num_logical: int) -> QuantumCircuit:
    """Undo a transpiler layout, so qubit ``i`` is the subcircuit's own qubit ``i``.

    Transpiling against a backend lays the circuit out on physical qubits and pads it to
    the device width, so the qubit at index ``i`` afterwards is generally not the one
    that was there before. Everything downstream reads measurement bits by position --
    ``_get_sub_expectation_values`` picks observable bits out by index -- so a layout
    that permutes the qubits silently attributes results to the wrong ones.

    Relabelling the wires is exact and free: the operations and their order do not
    change, only which index each sits on. The logical qubits are put where the layout
    says they end up, since that is where the observables are read, and any wire routing
    borrowed on the way is kept after them. Placeholders are unaffected either way,
    because a placeholder is inserted on whichever wire it is already sitting on, which
    is where its qubit is at that point in the circuit.

    Wires that end up carrying nothing at all -- the device padding -- are dropped.

    Args:
        circuit (QuantumCircuit): a transpiled circuit carrying a layout.
        num_logical (int): how many qubits it had before transpilation.

    Returns:
        QuantumCircuit: the same circuit, its own qubits first and in order.
    """
    layout = circuit.layout
    if layout is None:
        return circuit

    physical_for_logical = list(layout.final_index_layout())[:num_logical]
    busy = {
        circuit.find_bit(qubit).index
        for instruction in circuit.data
        for qubit in instruction.qubits
    }
    # The subcircuit's own qubits first, then anything routing borrowed, then nothing:
    # idle padding is left out entirely.
    borrowed = sorted(busy - set(physical_for_logical))
    order = physical_for_logical + borrowed
    new_for_old = {old: new for new, old in enumerate(order)}

    out = QuantumCircuit(len(order), name=circuit.name)
    for register in circuit.cregs:
        out.add_register(register)
    # Wires past this are ones routing borrowed. They carry no part of the subcircuit's
    # state at the end, so nothing downstream should measure them.
    out.metadata = dict(circuit.metadata or {})
    out.metadata["qcut_logical_qubits"] = num_logical

    for instruction in circuit.data:
        physical = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        out.append(
            instruction.operation,
            [out.qubits[new_for_old[index]] for index in physical],
            instruction.clbits,
        )
    return out


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
