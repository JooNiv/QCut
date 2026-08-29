from qiskit.quantum_info import SparsePauliOp


def all_z_paulis_for_subset(number_of_qubits: int, qubit_indices: list[int]) -> SparsePauliOp:
    """Return a list of all n-qubit Pauli Z operators."""
    paulis = []
    for i in range(2**len(qubit_indices)):
        pauli_str = ["I"] * number_of_qubits
        for j, qubit_index in enumerate(qubit_indices):
            ind = number_of_qubits - qubit_index - 1
            if (i >> j) & 1:
                pauli_str[ind] = "Z"
        paulis.append("".join(pauli_str))
    return SparsePauliOp(paulis[1:])

