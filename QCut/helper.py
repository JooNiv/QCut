"""Helper finctions for circuit knitting."""

import numpy as np
#from openqaoa.qaoa_components import Hamiltonian
from qiskit.quantum_info import PauliList


def isclose(a: float, b: float) -> bool:
    """Check if two floats equal-ish.

    Args:
    ----
        b: float to be compared
        a: float to be compared.

    Returns:
    -------
        bool: whether a and b close to each other or not.

    """
    tolerance = 0.1
    return abs(a-b) <= tolerance

#calculate relative error
def relative_error(actual: list, approx: list) -> list:
    """Calculate the relative error."""
    if(np.prod(actual) == 0):
        return abs(approx-actual)/(1+abs(actual))

    return abs(approx-actual)/(abs(actual))

"""def hamiltonian_to_strings(hamiltonian: Hamiltonian, length: int) -> dict:
    Convert openQAOA hamiltonian to a convenient form.
    terms = str(hamiltonian).split("+")
    new_string = []
    coefs = []
    const = 0

    for term in terms:
        indices = [int(i.strip("_{ }")) for i in term.split("Z")[1:]]

        new_string = "I"*length

        if len(indices) != 0:
            coefs.append(float(term[:4]))
            list_string = list(new_string)

            for i in indices:
                list_string[i] = "Z"

            new_string.append("".join(list_string)[::-1])

        else:
            const += float(term[:4])

    return {"paulis":PauliList(new_string), "coefs": coefs, "const": const}"""
