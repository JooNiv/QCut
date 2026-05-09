"""Helper gates for circuit knitting."""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Instruction, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass

# define the cut location marker
cut_op = QuantumCircuit(1, name="Cut")
cut_op = cut_op.to_instruction(label="Cut")
cut_op.definition = None


def cut() -> QuantumCircuit | Instruction:
    """Return a single qubit wire cut instruction."""
    return cut_op


cutCZ_op = QuantumCircuit(2, name="CutCZ")
cutCZ_op = cutCZ_op.to_instruction(label="CutCZ")
cutCZ_op.definition = None

cutSWAP_op = QuantumCircuit(2, name="CutSWAP")
cutSWAP_op = cutSWAP_op.to_instruction(label="CutSWAP")
cutSWAP_op.definition = None

cutISWAP_op = QuantumCircuit(2, name="CutISWAP")
cutISWAP_op = cutISWAP_op.to_instruction(label="CutISWAP")
cutISWAP_op.definition = None


def cutCZ() -> QuantumCircuit | Instruction:
    """Return a two qubit cutCZ gate instruction."""
    return cutCZ_op


# "u" (UGate = U(θ,φ,λ)) is Qiskit's universal single-qubit rotation — any single-qubit
# unitary decomposes to it. Override via cutGate(single_qubit_basis=...) or replace
# this list to match a backend's native single-qubit gates.
QPD_DECOMPOSITION_SQ_BASIS: list[str] = ["u"]

# Maps 2-qubit gate name -> cut instruction.
# Add here once the QPD terms are implemented in qpd.py and QPD_REGISTRY in
# qpd_operations.py is updated — both must be done together.
QPD_GATE_REGISTRY: dict[str, Instruction] = {
    "cz": cutCZ_op,
    "swap": cutSWAP_op,
    "iswap": cutISWAP_op,
}


class _ReplaceWithCutGates(TransformationPass):
    def __init__(self, registry: dict[str, Instruction]):
        self.registry = registry
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if node.op.name not in self.registry:
                continue
            cut_op = self.registry[node.op.name]
            mini_dag = DAGCircuit()
            qreg = QuantumRegister(node.op.num_qubits)
            mini_dag.add_qreg(qreg)
            mini_dag.apply_operation_back(cut_op, list(qreg))
            dag.substitute_node_with_dag(node, mini_dag)
        return dag


def cutGate(
    gate: Gate,
    control: int | list[int],
    target: int | list[int],
    single_qubit_basis: list[str] | None = None,
) -> dict:
    """Return a CutGate circuit decomposing the input gate with QPD cut markers.

    control and target together define the ordered physical qubit mapping for
    gate inputs 0..n-1: gate input i maps to physical qubit (controls + targets)[i].
    """
    controls = [control] if isinstance(control, int) else list(control)
    targets = [target] if isinstance(target, int) else list(target)
    all_qargs = controls + targets

    if gate.num_qubits < 2:
        raise ValueError("Input gate must have at least 2 qubits.")
    if len(all_qargs) != gate.num_qubits:
        raise ValueError(
            f"Expected {gate.num_qubits} qubit arguments, got {len(all_qargs)}."
        )
    if len(set(all_qargs)) != len(all_qargs):
        raise ValueError("Qubit arguments must be unique.")
    if any(q < 0 for q in all_qargs):
        raise ValueError("Qubit arguments must be non-negative.")

    qc = QuantumCircuit(gate.num_qubits, name=f"cut{gate.name.upper()}")
    qc.append(gate, list(range(gate.num_qubits)))

    sq_basis = single_qubit_basis if single_qubit_basis is not None else QPD_DECOMPOSITION_SQ_BASIS
    tr = transpile(qc, basis_gates=sq_basis + list(QPD_GATE_REGISTRY.keys()), optimization_level=0)
    tr = PassManager([_ReplaceWithCutGates(QPD_GATE_REGISTRY)]).run(tr)

    return {
        "instruction": tr.to_instruction(label="CutGate"),
        "qargs": all_qargs,
    }


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
