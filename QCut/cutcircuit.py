"""Class for nicely representing a cut circuit. Also implements some of the same
functionality as the qiskit QuantumCircuit class for a group of circuts."""

from qiskit import QuantumCircuit

from QCut.cutlocation import CutLocation, SingleQubitCutLocation
    

class CutCircuit:
    def __init__(self, 
                    subcircuits: dict[int, QuantumCircuit],
                    cut_locations: list[CutLocation | SingleQubitCutLocation],
                    map_qubit: dict[int, int],
                    backend=None) -> None:
        """Init."""

        self.subcircuits = subcircuits
        self.cut_locations = cut_locations
        self.map_qubit = map_qubit
        self.backend = backend

    def assign_parameters(self, parameters: dict, inplace=False
                          ) -> dict[int, QuantumCircuit]:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters."""
        if inplace:
            for ind, circuit in enumerate(self.subcircuits):
                try:
                    self.subcircuits[ind] = circuit.assign_parameters(parameters)
                except Exception:
                    pass
            #return self.subcircuits
        
        else:
            new_circuits = {}
            for ind, circuit in enumerate(self.subcircuits):
                try:
                    new_circuits[ind] = circuit.assign_parameters(parameters)
                except Exception:
                    new_circuits[ind] = circuit
            return CutCircuit(new_circuits,
                              self.cut_locations, self.map_qubit, self.backend)
            
    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.subcircuits]
    
    @property
    def num_subcircuits(self):
        """Total number of circuits."""
        return len(self.subcircuits)


class CutExperiment:
    def __init__(self, 
                    experiment_circuits: list[list[dict[int, QuantumCircuit]]],
                    cut_locations: list[CutLocation | SingleQubitCutLocation],
                    map_qubit: dict[int, int],
                    coefficients: list[float],
                    observables,
                    backend=None) -> None:
        """Init."""

        self.experiments = experiment_circuits
        self.backend = backend
        self.cut_locations = cut_locations
        self.map_qubit = map_qubit
        self.coefficients = coefficients
        self.observables = observables

    def expv_data(self):
        """Get data for expv calculation."""
        return {
            "cut_locations": self.cut_locations,
            "map_qubit": self.map_qubit,
            "coefficients": self.coefficients,
            "observables": self.observables
        }

    def assign_parameters(self, parameters: dict, inplace=False
                          ) -> list[dict[int, QuantumCircuit]]:
        """Assign parameters to the circuits. Same as qiskit
        QuantumCircuit.assign_parameters."""
        if inplace:
            for exp_ind, subcircuits in enumerate(self.experiments):
                for circ_ind, value in enumerate(subcircuits):
                    for ind, circuit in value.items():
                        try:
                            self.experiments[exp_ind][circ_ind][ind] = (circuit
                                                            .assign_parameters(parameters))
                        except Exception:
                            pass
            #return self.experiments
        else:
            new_experiments = []
            for exp_ind, subcircuits in enumerate(self.experiments):
                new_subcircuits = []
                for circ_ind, value in enumerate(subcircuits):
                    new_circuits = {}
                    for ind, circuit in value.items():
                        try:
                            new_circuits[ind] = (circuit
                                                .assign_parameters(parameters))
                        except Exception:
                            new_circuits[ind] = circuit
                    new_subcircuits.append(new_circuits)
                new_experiments.append(new_subcircuits)
            return CutExperiment(new_experiments,
                                 self.cut_locations, self.map_qubit,
                                 self.coefficients, self.observables, self.backend)
    
    @property
    def num_qubits(self):
        """Number of qubits per subcircuit."""
        return [i.num_qubits for i in self.experiments[0][0].values()]
    
    @property
    def num_circuits(self):
        """Total number of circuits."""
        return sum(len(subcircuits) for subcircuits in self.experiments)
    
    @property
    def group_size(self):
        """Number of circuits in a group."""
        return len(self.experiments[0][0][0])
    
    @property
    def num_groups(self):
        """Number of circuit groups."""
        return len(self.experiments)
    
    @property
    def num_obs_groups(self):
        return len(self.experiments[0])