"""Helper functions for circuit knitting."""

import numpy as np
from qiskit.quantum_info import PauliList


def isclose(a: float, b: float) -> bool:
    """Check if two floats equal-ish.

    Args:
    -----
        b: float to be compared
        a: float to be compared.

    Returns:
    --------
        bool: whether a and b close to each other or not.

    """
    tolerance = 0.1
    return abs(a - b) <= tolerance


# calculate relative error
def relative_error(actual: list, approx: list) -> list:
    """Calculate the relative error."""
    if np.prod(actual) == 0:
        return abs(approx - actual) / (1 + abs(actual))

    return abs(approx - actual) / (abs(actual))


def get_pauli_list(input_list: list, length: int) -> PauliList:
    """Transform list of observable indices to Paulilist of Z observables.

    Args:
    -----
        input_list: lits of observables as qubit indices
        length: number of qubits in the circuit

    Returns:
    --------
        PauliList: a PauliList of Z observables

    """
    result = []
    base_string = "I" * length

    for indices in input_list:
        temp_string = list(base_string)
        if isinstance(indices, int):
            temp_string[indices] = "Z"
        else:
            for index in indices:
                temp_string[index] = "Z"
        result.append("".join(temp_string))

    return PauliList(result)
