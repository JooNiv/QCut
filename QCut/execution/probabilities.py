from qiskit.quantum_info import SparsePauliOp

from QCut.execution.postprocess import estimate_expectation_values
from QCut.execution.qcutresult import RawResult


def all_z_paulis_for_subset(number_of_qubits: int, 
                            qubit_indices: list[int]) -> SparsePauliOp:
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

def gen_bitstrings(n):
    """Generate all bitstrings of length n."""
    return [format(i, f'0{n}b') for i in range(2**n)]

def gen_subsets(n):
    """Generate all subsets of a set of size n."""
    subsets = []
    for i in range(2**n):
        subset = [j for j in range(n) if (i & (1 << j))]
        subsets.append(subset)
    return subsets

def map_expvs_to_subsets(expvs, subsets):
    """Map expectation values to their corresponding subsets."""
    expv_mapping = {(): 1.0}  # Initialize with the identity observable
    subsets_no_empty = subsets[1:]  # Exclude the empty subset
    for subset, expv in zip(subsets_no_empty, expvs):
        expv_mapping[tuple(subset)] = expv
    return expv_mapping

def parity_bits(bitstring, subset):
    """Calculate the parity of a bitstring for a given subset of indices."""
    bits = bitstring[::-1]
    return sum(int(bits[i]) for i in subset) % 2

def reconstruct_probs(expv_subset_mapping, bits):
    """Reconstruct the probabilities of each bitstring from the expectation values."""
    probs = {}
    for x in bits:
        total = sum((-1)**parity_bits(x, subset) * expv 
                    for subset, expv in expv_subset_mapping.items())
        probs[x] = total / 2**len(bits[0])
    return probs

def estimate_probabilities(result: RawResult) -> dict[str, float]:
    exps = estimate_expectation_values(result)
    label = result.experiment.observables.paulis[-1].to_label()
    filtered = ''.join([char for char in label if char == 'Z'])

    len_subset = len(filtered)

    print(f"Estimating probabilities for {len_subset} qubits.")
    
    subsets = gen_subsets(len_subset)

    print(f"Generated {len(subsets)} subsets for {len_subset} qubits.")

    expv_mapping = map_expvs_to_subsets(exps, subsets)

    print(f"Mapped expectation values to {len(expv_mapping)} subsets.")

    bits = gen_bitstrings(len_subset)

    print(f"Generated {len(bits)} bitstrings for {len_subset} qubits.")

    return reconstruct_probs(expv_mapping, bits)