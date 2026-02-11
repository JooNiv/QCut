import QCut as ck
from QCut import cut, cutGate
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import CXGate
from qiskit.circuit import Parameter

cut_qc = QuantumCircuit(4)

mult = Parameter('mult')
cut_qc.r(mult*0.46262, mult*0.1446, 0)
cut_qc.append(**cutGate(CXGate(), 0, 1)) 
cut_qc.append(cut, [1])
cut_qc.cx(1,2)
cut_qc.cx(2,3)


cut_circuit = ck.get_locations_and_subcircuits(cut_qc)
num_subcircuits = 3

num_qubits = [1,1,3]

def test_cut_circuit_properties():
    assert cut_circuit.num_qubits == num_qubits
    assert cut_circuit.num_subcircuits == num_subcircuits

def test_cut_circuit_assign_parameters():
    cut_circuit_param = cut_circuit.assign_parameters({'mult': 2}, inplace=False)

    for exp_ind, subcircuits in enumerate(cut_circuit_param.subcircuits):
        for circ_ind, value in enumerate(subcircuits):
            for par in value.params:
                if hasattr(par, 'parameters'):
                    for elem in par.parameters:
                        assert 'mult' not in elem.name
                else:
                    assert True

observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])
cut_experiment = ck.get_experiment_circuits(cut_circuit, observables)

exp_num_qubits = [1,1,3]
exp_num_circuits = 144
exp_group_size = 3
exp_num_groups = 48
exp_num_obs_groups = 1

def test_cut_experiment_properties():
    assert cut_experiment.num_qubits == exp_num_qubits
    assert cut_experiment.num_circuits == exp_num_circuits
    assert cut_experiment.group_size == exp_group_size
    assert cut_experiment.num_groups == exp_num_groups
    assert cut_experiment.num_obs_groups == exp_num_obs_groups

def test_cut_experiment_assign_parameters():
    cut_experiment_param = cut_experiment.assign_parameters({'mult': 2}, inplace=False)

    for exp_ind, subcircuits in enumerate(cut_experiment_param.experiments):
        for obs_circ in subcircuits:
            for circ in obs_circ.values():
                for circ_ind, value in enumerate(circ):
                    for par in value.params:
                        if hasattr(par, 'parameters'):
                            for elem in par.parameters:
                                assert 'mult' not in elem.name
                        else:
                            assert True

   
