from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Instruction, Qubit

from QCut import cut
from QCut.cutlocation import SingleQubitCutLocation as CutLocation

qc_1 = QuantumCircuit(3)
qc_1.x(0)
qc_1.cx(0,1)
qc_1.append(cut, [1])
qc_1.cx(1,2)

qc_2 = QuantumCircuit(4)
qc_2.x(0)
qc_2.cx(0,1)
qc_2.cx(0,2)
qc_2.append(cut, [1])
qc_2.append(cut, [2])
qc_2.cx(1,3)
qc_2.cz(2,3)

qc_3 = QuantumCircuit(4)
qc_3.x(0)
qc_3.cx(0,1)
qc_3.append(cut, [1])
qc_3.cx(1,2)
qc_3.append(cut, [2])
qc_3.cz(2,3)

qc_4 = QuantumCircuit(5)
qc_4.h(0)
qc_4.cx(1,2)
qc_4.cx(2,3)
qc_4.append(cut, [1])
qc_4.append(cut, [3])
qc_4.cx(0,1)
qc_4.cx(3,4)
qc_4.cx(0,4)

qc_5 = QuantumCircuit(4)
qc_5.h(0)
qc_5.cx(1,2)
qc_5.append(cut, [1])
qc_5.append(cut, [2])
qc_5.cx(0,1)
qc_5.cx(2,3)
qc_5.cx(0,3)

test_circuits = [qc_1, qc_2, qc_3, qc_4, qc_5]

cut_location_solutions = [[CutLocation(((QuantumRegister(3, "q"), 1), 2))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 3)), CutLocation(((QuantumRegister(4, "q"), 2), 3))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 2)), CutLocation(((QuantumRegister(4, "q"), 2), 3))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 3)), CutLocation(((QuantumRegister(4, "q"), 3), 3))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 2)), CutLocation(((QuantumRegister(4, "q"), 2), 2))],
                        ]


subcircuit_solutions = [
    [
        [CircuitInstruction(operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 1),), clbits=())],
        [CircuitInstruction(operation=Instruction(name='Init_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=())]
    ],
    [
        [CircuitInstruction(operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0), Qubit(QuantumRegister(3, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 1),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0), Qubit(QuantumRegister(3, 'q'), 2)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 2),), clbits=())],
        [CircuitInstruction(operation=Instruction(name='Init_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='Init_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 1),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0), Qubit(QuantumRegister(3, 'q'), 2)), clbits=()), CircuitInstruction(operation=Instruction(name='cz', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 1), Qubit(QuantumRegister(3, 'q'), 2)), clbits=())]
    ],
    [
        [CircuitInstruction(operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 1),), clbits=())],
        [CircuitInstruction(operation=Instruction(name='Init_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 1),), clbits=())],
        [CircuitInstruction(operation=Instruction(name='Init_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cz', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=())]
    ],
    [
        [CircuitInstruction(operation=Instruction(name='h', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='Init_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 1),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0), Qubit(QuantumRegister(4, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Init_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 2),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 2), Qubit(QuantumRegister(4, 'q'), 3)), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0), Qubit(QuantumRegister(4, 'q'), 3)), clbits=())] ,
        [CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0), Qubit(QuantumRegister(3, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 1), Qubit(QuantumRegister(3, 'q'), 2)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(3, 'q'), 2),), clbits=())] ,

    ],
    [
        [CircuitInstruction(operation=Instruction(name='h', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='Init_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 1),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0), Qubit(QuantumRegister(4, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Init_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 2),), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 2), Qubit(QuantumRegister(4, 'q'), 3)), clbits=()), CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 0), Qubit(QuantumRegister(4, 'q'), 3)), clbits=()), CircuitInstruction(operation=Instruction(name='obs_2', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(4, 'q'), 2),), clbits=())] ,
        [CircuitInstruction(operation=Instruction(name='cx', num_qubits=2, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0), Qubit(QuantumRegister(2, 'q'), 1)), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_0', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),), clbits=()), CircuitInstruction(operation=Instruction(name='Meas_1', num_qubits=1, num_clbits=0, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 1),), clbits=())] ,
    ],
]

test_observables = [
    [0,1,2, [0,2], [1,2], [0,1,2]],
    [0,1,2,3, [0,3], [1,2,3], [0,1,2,3]],
    [0,1,2,3, [0,3], [1,2,3], [0,1,2,3]],
    [0,1,2, [0,2], [0,3,4]],
    [0,1,2,3, [0,2], [0,3,1]],
]

exp_val_solutions = [
    [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
]