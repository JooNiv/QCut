from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import cut, cutGate

circuit  =  QuantumCircuit(4)

mult = 1.635
circuit.r(mult*0.46262, mult*0.1446, 0)
circuit.cx(0,1)
circuit.cx(1,2)
circuit.cx(2,3)
   

cut_circuit = QuantumCircuit(4)

mult = 1.635
cut_circuit.r(mult*0.46262, mult*0.1446, 0)
cut_circuit.append(**cutGate(CXGate(), 0, 1)) 
cut_circuit.append(cut(), [1])
cut_circuit.cx(1,2)
cut_circuit.cx(2,3)

subcirc_lens = [3,4,6]

res_expvs = [0.727323, 0.727323, 0.727323, 1.000000]

def test_cut_gate_num_subcircuits():
    cut_circ = cut_circuit.copy()

    cut_qc = ck.get_locations_and_subcircuits(cut_circ)

    assert cut_qc.num_subcircuits == 3

def test_cut_gate_subcircuits():
    cut_qc = ck.get_locations_and_subcircuits(cut_circuit.copy())

    for ind, circ in enumerate(cut_qc.subcircuits):
        assert len(circ.data) == subcirc_lens[ind]

def test_cut_gate_expectation_values():
    cut_qc = ck.get_locations_and_subcircuits(cut_circuit.copy())

    backend = AerSimulator()

    observables = SparsePauliOp(["IIIZ", "IIZI", "IZII", "IIZZ"])

    cut_experiment = ck.get_experiment_circuits(cut_qc, observables)

    results = ck.run_experiments(cut_experiment, backend=backend)

    expectation_values = ck.estimate_expectation_values(results, 
                                                        cut_experiment.expv_data())

    for ind, expv in enumerate(expectation_values):
        assert abs(expv - res_expvs[ind]) < 0.1