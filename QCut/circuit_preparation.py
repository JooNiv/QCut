"""
A module for preparing the circuit for experiment generation and the proper
circuit knitting workflow, including finding cut locations,
inserting placeholder operations, and separating into subcircuits.
"""

from __future__ import annotations

import logging

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Instruction, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from QCut.consolidate import consolidate_two_qubit_blocks, marker_gate
from QCut.cutcircuit import CutCircuit
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.options import CutOptions, resolve
from QCut.qcuterror import QCutError

logger: logging.Logger = logging.getLogger(__name__)


def _get_cut_locations(circuit):
    index = 0  # index of the current instruction in circuit_data
    circuit_data = circuit.data
    cut_locations = []

    # loop through circuit instructions
    # if operation is a Cut() instruction remove it and add registers and
    # offset index to cut_locations

    # rename varibales to be more descriptive (namely qs)
    while index < len(circuit):
        op = circuit_data[index]
        if "Cut" in op.operation.name:
            # find qubits for Cut operation
            qubits = [circuit.find_bit(qubit).registers[0] for qubit in op.qubits]

            # remove the cut operation
            circuit_data.remove(op)

            # append to cut_locations
            if len(qubits) == 1:
                cut_locations.append(SingleQubitCutLocation((qubits[0], index)))
            else:
                gate_name = op.operation.name.replace("Cut", "").lower()
                # CutTwoQubitGate markers carry the gate so a QPD can be generated
                # from it. The per-gate markers do not need to.
                cut_locations.append(
                    CutLocation(
                        (qubits, index),
                        gate_name=gate_name,
                        gate=getattr(op.operation, "gate", None),
                    )
                )

            # adjust index to account for removed operation
            index -= 1
        index += 1

    return cut_locations


class NonCommutingGate(Instruction):
    def __init__(self, name="Init_1"):
        super().__init__(name=name, num_qubits=1, num_clbits=0, params=[])
        self._opaque = True

    def __repr__(self):
        return f"{self.name}"


def _insert_cut_nodes(circuit, cut_locations):
    circuit_data = circuit.data
    cut_index = 0
    offset = 0
    for cut_location in cut_locations:
        measure_node = NonCommutingGate(f"Meas_{cut_index}")

        initialize_node = NonCommutingGate(f"Init_{cut_index}")

        prefix = (
            f"cut{cut_location.gate_name.upper()}"
            if isinstance(cut_location, CutLocation)
            else ""
        )
        cut_czc = NonCommutingGate(f"{prefix}_c_{cut_index}")

        cut_czt = NonCommutingGate(f"{prefix}_t_{cut_index}")

        cut_index += 1

        cur_ops = (
            (measure_node, initialize_node)
            if isinstance(cut_location, SingleQubitCutLocation)
            else (cut_czc, cut_czt)
        )

        if isinstance(cut_location, SingleQubitCutLocation):
            for ph_op in cur_ops:
                circuit_data.insert(
                    cut_location.index + offset,
                    CircuitInstruction(
                        operation=ph_op,
                        qubits=[Qubit(cut_location.qubits[0], cut_location.qubits[1])],
                    ),
                )

                offset += 1

        else:
            circuit_data.insert(
                cut_location.index + offset,
                CircuitInstruction(
                    operation=cur_ops[0],
                    qubits=[
                        Qubit(cut_location.qubits[0][0], cut_location.qubits[0][1])
                    ],
                ),
            )

            offset += 1

            circuit_data.insert(
                cut_location.index + offset,
                CircuitInstruction(
                    operation=cur_ops[1],
                    qubits=[
                        Qubit(cut_location.qubits[1][0], cut_location.qubits[1][1])
                    ],
                ),
            )

            offset += 1

    return circuit


def _move_to_new_wire(orig: QuantumCircuit) -> QuantumCircuit:
    # Create the new circuit and add registers
    offset = 0
    new = QuantumCircuit(name=orig.name + "_rebuilt")
    for creg in orig.cregs:
        new.add_register(ClassicalRegister(creg.size, creg.name))
    for qreg in orig.qregs:
        new.add_register(QuantumRegister(0, qreg.name))

    # Create fresh Qubit objects for each original wire
    # and record a mapping old_qubit -> new_qubit
    qubit_map = {}
    new_qubits = []
    for idx, q_old in enumerate(orig.qubits):
        q_new = Qubit()
        new_qubits.append(q_new)
        qubit_map[q_old] = q_new
    new.add_bits(new_qubits)

    # 3) Replay every instruction, splitting on Measure
    for inst, qargs, cargs in orig.data:
        # map every qarg via our current mapping
        mapped_qs = [qubit_map[q] for q in qargs]

        if inst.name.startswith("Meas"):
            # append this measurement on the current wire
            new.append(inst, mapped_qs, cargs)

            # now allocate a fresh wire for all future uses of qargs[0]
            q_fresh = Qubit()
            new.add_bits([q_fresh])
            new.qubits.remove(q_fresh)
            new.qubits.insert(orig.find_bit(qargs[0]).index + 1 + offset, q_fresh)
            offset += 1
            # update the mapping so q_old -> q_fresh going forward
            qubit_map[qargs[0]] = q_fresh

        else:
            # just copy the gate over to mapped_qs
            new.append(inst, mapped_qs, cargs)

    return new


def _separate_subcircuits(circuit):
    dag = circuit_to_dag(circuit)

    circs = dag.separable_circuits(remove_idle_qubits=True)

    new_circs = []
    for i in circs:
        circ = dag_to_circuit(i)
        if len(circ.qubits) == 0:
            continue
        new_circs.append(circ)

    return new_circs


def _add_cbits(subcircuits):
    for circ in subcircuits:
        clbits = 0
        clbits_qpd = 0
        for i in circ:
            name = i.operation.name
            if "Meas" in name:
                clbits_qpd += 1
            elif "cut" in name:
                clbits += 1
                clbits_qpd += 1

        circ.add_register(ClassicalRegister(clbits_qpd, "qpd_meas"))
        circ.add_register(
            ClassicalRegister(circ.num_qubits - clbits_qpd + clbits, "meas")
        )

    return subcircuits


def get_qubit_map(subcircuits: list[QuantumCircuit]):
    def filter_obs_i(qc_data):
        return [i for i in qc_data if "obs" in i.operation.name]

    def sort_func(obs):
        return int(obs.operation.name.split("_")[1])

    map_qubit = {}
    count = 0
    for ind, i in enumerate(reversed(subcircuits)):
        for j in sorted(
            filter_obs_i(i.data),
            key=lambda x: i.find_bit(x.qubits[0]).index,
            reverse=True,
        ):
            map_qubit[int(j.operation.name.split("_")[1])] = count
            count += 1

    return map_qubit


def _consolidate_marked_pairs(circuit: QuantumCircuit) -> QuantumCircuit:
    """Merge runs of gates on qubit pairs that carry a cut marker.

    Pairs with no marker are left alone. Merging them would not change any cutting cost
    and would only replace named gates by generic unitaries.
    """
    marked = {
        frozenset(circuit.find_bit(q).index for q in instruction.qubits)
        for instruction in circuit.data
        if marker_gate(instruction.operation) is not None
    }
    if not marked:
        return circuit
    return consolidate_two_qubit_blocks(circuit, restrict_to=marked)


def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
    max_qubits: list[int] | None = None,
    options: CutOptions | None = None,
) -> CutCircuit:
    """Get cut locations and subcircuits with placeholder operations.

    Args:
        circuit (QuantumCircuit): circuit with cuts inserted
        max_qubits (list[int], optional):
            list of maximum qubits per subcircuit when using automatic cut
            finding. If None, no constraint is used. Defaults to None.
            In general it is not necesary to manually specify this parameter.
        options (CutOptions, optional): configuration for the run. Defaults to
            QCut.options.DEFAULT_OPTIONS.

    Returns:
        tuple: A tuple containing:
            - list[SingleQubitCutLocation]: Locations of the cuts as a list
            - list[QuantumCircuit]: Subcircuits with placeholder operations
            - dict[int:int]: map of subcircuit qubit indices to original circuit
                            qubit indices

    """
    from QCut.QCutFind import construct_final_subcircuits

    options = resolve(options)
    circuit_copy = circuit.copy()  # copy to avoid modifying the original circuit
    circuit_copy = circuit_copy.decompose(["CutGate"])
    if options.consolidate:
        # Before the obs_i tags go on, since those touch every qubit and would end every
        # run. Cut locations are recorded afterwards, so merged runs count as one cut.
        circuit_copy = _consolidate_marked_pairs(circuit_copy)
    for i in range(circuit.num_qubits):
        obs_m = QuantumCircuit(1, name=f"obs_{i}")
        obs_m = obs_m.to_instruction()
        circuit_copy.append(obs_m, [i])
    cut_locations = _get_cut_locations(circuit_copy)
    circuit1 = _insert_cut_nodes(circuit_copy, cut_locations)
    circuit_new = _move_to_new_wire(circuit1.copy())
    subcircuits = _separate_subcircuits(circuit_new)

    subcircuits = _add_cbits(subcircuits)
    fixed_circs = []
    for i in subcircuits:
        test = QuantumCircuit(i.num_qubits)
        test.add_register(i.cregs[0])
        test.add_register(i.cregs[1])

        for j in i.data:
            qubits = [test.qubits[i.qubits.index(q)] for q in j.qubits]
            test.append(CircuitInstruction(j.operation, qubits))

        fixed_circs.append(test)
    if len(fixed_circs) <= 1:
        raise QCutError(
            "Invalid cuts. Check documentation to see how cuts should be placed."
        )

    if max_qubits and len(fixed_circs) != len(max_qubits):
        """if max_qubits is None:
            raise QCutError(
                "max_qubits must be specified when automatic cut finding with " \
                "max_qubits constraint is used."
            )"""
        fixed_circs = construct_final_subcircuits(fixed_circs, max_qubits)

    map_qubits = get_qubit_map(fixed_circs)

    num_wire_cuts = len(
        [loc for loc in cut_locations if isinstance(loc, SingleQubitCutLocation)]
    )
    num_gate_cuts = len([loc for loc in cut_locations if isinstance(loc, CutLocation)])

    logger.info(
        f"Found {len(cut_locations)} cut locations"
        f"({num_wire_cuts})"
        f" wire cut(s) and {num_gate_cuts} gate cut(s))"
        f" and separated into {len(fixed_circs)} subcircuits."
    )

    return CutCircuit(fixed_circs, cut_locations, map_qubits, options=options)
