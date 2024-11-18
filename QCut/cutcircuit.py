"""Class for nicely representing a cut circuit. Also implements some of the same
functionality as the qiskit QuantumCircuit class for a group of circuts."""

from qiskit import QuantumCircuit


class CutCircuit:
    def __init__(self, experiment_circuits: list[list[QuantumCircuit]]) -> None:
        """Init."""
        self.circuits = experiment_circuits

    def assign_parameters(self, parameters: dict) -> list[list[QuantumCircuit]]:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters."""
        new_circuits = [[0 for _ in range(len(row))] for row in self.circuits]
        for ind, group in enumerate(self.circuits):
            for ind_2, circuit in enumerate(group):
                try:
                    new_circuits[ind][ind_2] = circuit.assign_parameters(parameters)
                except Exception:
                    new_circuits[ind][ind_2] = circuit
        print(new_circuits[0])
        return new_circuits
    
    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.circuits[0]]
    
    @property
    def num_circuits(self):
        """Total number of circuits."""
        return len(self.circuits[0]*len(self.circuits))
    
    @property
    def group_size(self):
        """Number of circuits in a group."""
        return len(self.circuits[0])
    
    @property
    def num_groups(self):
        """Number of circuit groups."""
        return len(self.circuits)