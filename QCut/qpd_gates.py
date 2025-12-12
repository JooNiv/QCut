"""Helper gates for circuit knitting."""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import CircuitInstruction, Gate

# define the cut location marker
cut = QuantumCircuit(1, name="Cut")
cut = cut.to_instruction(label="Cut")
cut.definition = None

cutCZ = QuantumCircuit(2, name="CutCZ")
cutCZ = cutCZ.to_instruction(label="CutCZ")
cutCZ.definition = None

def cutGate(gate: Gate, control: int, target: int) -> dict:
    """Return a cutCZ circuit with the same parameters as the input gate."""
    if gate.num_qubits != 2:
        raise ValueError("Input gate must be a 2-qubit gate.")
    if control == target:
        raise ValueError("Control and target qubits must be different.")
    if control < 0 or target < 0:
        raise ValueError("Control and target qubits must be non-negative.")

    qc = QuantumCircuit(2, name=f"cut{gate.name.upper()}")
    if control > target:
        loccontrol = 1
        loctarget = 0
    else:
        loccontrol = 0
        loctarget = 1
    qc.append(gate, [loccontrol, loctarget])
    tr = transpile(qc, basis_gates=["cz", "r", "h", "s", "sdg", "x", "y", "z"])
    for ind, instr in enumerate(tr.data):
        if instr.operation.name == "cz":
            tr.data.pop(ind)
            test = CircuitInstruction(
                        operation=cutCZ,
                        qubits=tr.qubits,
                    )
            tr.data.insert(ind, test)
    return {"instruction": tr.to_instruction(label="CutGate"), 
            "qargs":[control, target]}

# define measurements for different bases
xmeas = QuantumCircuit(1, 1, name="x-meas")
xmeas.h(0)
xmeas.measure(0, 0)

ymeas = QuantumCircuit(1, 1, name="y-meas")
ymeas.sdg(0)
ymeas.h(0)
ymeas.measure(0, 0)

idmeas = QuantumCircuit(1, name="id-meas")
idmeas.id(0)

zmeas = QuantumCircuit(1, 1, name="z-meas")
zmeas.measure(0, 0)


# define initialization operations
zero_init = QuantumCircuit(1, name="0-init")
zero_init.id(0)

one_init = QuantumCircuit(1, name="1-init")
one_init.x(0)

plus_init = QuantumCircuit(1, name="'+'-init")
plus_init.h(0)

minus_init = QuantumCircuit(1, name="'-'-init")
minus_init.h(0)
minus_init.z(0)

i_plus_init = QuantumCircuit(1, name="'i+'-init")
i_plus_init.h(0)
i_plus_init.s(0)

i_minus_init = QuantumCircuit(1, name="'i-'-init")
i_minus_init.h(0)
i_minus_init.z(0)
i_minus_init.s(0)

sdg = QuantumCircuit(1, 1, name="sdg")
sdg.sdg(0)

s = QuantumCircuit(1, 1, name="s")
s.s(0)

z = QuantumCircuit(1, 1, name="z")
z.z(0)

sdg_meas = QuantumCircuit(1, 1, name="sdg_meas")
sdg_meas.sdg(0)
sdg_meas.measure(0, 0)