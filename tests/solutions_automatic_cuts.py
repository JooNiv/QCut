from qiskit import QuantumCircuit

qc_1 = QuantumCircuit(3)
qc_1.x(0)
qc_1.cx(0,1)
qc_1.cx(1,2)

qc_2 = QuantumCircuit(4)
qc_2.x(0)
qc_2.cx(0,1)
qc_2.cx(0,2)
qc_2.cx(1,3)
qc_2.cz(2,3)

qc_3 = QuantumCircuit(4)
qc_3.x(0)
qc_3.cx(0,1)
qc_3.cx(1,2)
qc_3.cz(2,3)

qc_4 = QuantumCircuit(5)
qc_4.h(0)
qc_4.cx(1,2)
qc_4.cx(2,3)
qc_4.cx(0,1)
qc_4.cx(3,4)
qc_4.cx(0,4)

qc_5 = QuantumCircuit(4)
qc_5.h(0)
qc_5.cx(1,2)
qc_5.cx(0,1)
qc_5.cx(2,3)
qc_5.cx(0,3)

qc_6 = QuantumCircuit(3)
qc_6.h(0)
qc_6.cx(0, 1)
qc_6.cx(1, 2)
qc_6.cx(0, 1)

test_circuits = [qc_1, qc_2, qc_3, qc_4, qc_5, qc_6]

cut_sizes = [2,2,3,2,2,2]

test_observables = [
    [0,1,2, [0,2], [1,2], [0,1,2]],
    [0,1,2,3, [0,3], [1,2,3], [0,1,2,3]],
    [0,1,2,3, [0,3], [1,2,3], [0,1,2,3]],
    [0,1,2, [0,2], [0,3,4]],
    [0,1,2,3, [0,2], [0,3,1]],
    [0, 1, 2, [0, 2], [0, 1]],
]

exp_val_solutions = [
    [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 0.0],
]