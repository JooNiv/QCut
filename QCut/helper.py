"""Helper functions for circuit knitting."""

import numpy as np
from qiskit.quantum_info import PauliList


# calculate relative error
def relative_error(actual: list, approx: list) -> list:
    """
    Calculate the relative error between actual and approximate values.
    Args:
        actual (list): The list of actual values.
        approx (list): The list of approximate values.
    Returns:
        list:
            The list of relative errors for each corresponding pair of actual
            and approximate values.
    Raises:
        ValueError: If the lengths of actual and approx lists are not the same.
    """
    if np.prod(actual) == 0:
        return abs(approx - actual) / (1 + abs(actual))

    return abs(approx - actual) / (abs(actual))


def get_pauli_list(input_list: list, length: int) -> PauliList:
    """Transform list of observable indices to Paulilist of Z observables.

    Args:
        input_list (list): list of observables as qubit indices
        length (int): number of qubits in the circuit

    Returns:
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
