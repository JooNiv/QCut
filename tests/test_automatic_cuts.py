"""Tests for CircuitKnitting package."""  # noqa: N999

from qiskit_aer import AerSimulator

#import QCut as ck
import QCut.single_qubit_wirecut as wc
import tests.solutions_automatic_cuts as sq
from QCut import find_cuts


def test_find_cuts() -> None:
    """Test find_cuts function.

    This function tests whether the find_cuts method correctly identifies the cut
    locations in the provided test circuits by comparing the result to the
    pre-defined solutions.
    """
    for solution_index, circ in enumerate(sq.test_circuits):

        cut_locations, subcircuits, map_qubit = find_cuts(circ.copy(), 
                                                          sq.cut_sizes[solution_index], 
                                                          cuts="both")

        assert len(subcircuits) == sq.cut_sizes[solution_index]
        

def test_expectation_values() -> None:
    """Test the expectation values of the test circuits.

    This function tests whether the run method correctly calculates the expectation
    values for each test circuit and its corresponding observable by comparing the
    results to the pre-defined solutions within a specified error tolerance.

    The test runs each circuit on the AerSimulator backend without error mitigation.
    """
    # Initialize the simulator
    sim = AerSimulator()

    # Iterate over each test circuit and its corresponding expected solutions
    for solution_index, circ in enumerate(sq.test_circuits):
        print(solution_index)

        cut_locations, subcircuits, map_qubit = find_cuts(circ.copy(), 
                                                          sq.cut_sizes[solution_index], 
                                                          cuts="both")

        # Calculate expectation values using the run method
        estimated_expectation_values = wc.run_cut_circuit(subcircuits, cut_locations, 
                                                          sq.test_observables[solution_index], 
                                                          map_qubit, sim)
        # Check each calculated expectation value against the corresponding
        # expected value
        tolerance = 0.1
        print(estimated_expectation_values)
        print(sq.exp_val_solutions[solution_index])
        for check in [
            abs(a - b) <= tolerance
            for a, b in zip(estimated_expectation_values, 
                            sq.exp_val_solutions[solution_index])
        ]:
            assert check  # noqa: S101
