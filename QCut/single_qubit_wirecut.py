from __future__ import annotations

from collections import namedtuple
from copy import deepcopy

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Instruction, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator

from QCut.backend_utility import transpile_subcircuits
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.qcuterror import QCutError
from QCut.QCutFind import construct_final_subcircuits
from QCut.wirecut import (
    estimate_expectation_values,
    get_experiment_circuits,
    run_experiments,
)

BitLocations = namedtuple("BitLocations", ("index", "registers"))


def _get_cut_locations(circuit):
    index = 0  # index of the current instruction in circuit_data
    circuit_data = circuit.data
    cut_locations = np.array([])

    # loop through circuit instructions
    # if operation is a Cut() instruction remove it and add registers and
    # offset index to cut_locations

    # rename varibales to be more descriptive (namely qs)
    while index < len(circuit):
        op = circuit_data[index]
        if "Cut" in op.operation.name:
            # find qubits for Cut operation
            qubits = [
                circuit.find_bit(qubit).registers[0]
                for qubit in op.qubits
            ]

            # remove the cut operation
            circuit_data.remove(op)

            # append to cut_locations
            if len(qubits) == 1:
                cut_locations = np.append(
                    cut_locations, SingleQubitCutLocation((qubits[0], index))
                )
            elif len(qubits) == 2:
                cut_locations = np.append(
                    cut_locations, CutLocation((qubits, index))
                )
            else:
                raise QCutError("Cannot cut gates with more that 2 qubits." \
                "Transpile circuit to only contain 2 qubit gates.")

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

        cut_czc = NonCommutingGate(f"cutCZ_c_{cut_index}")
        
        cut_czt = NonCommutingGate(f"cutCZ_t_{cut_index}")

        cut_index += 1

        cur_ops = (measure_node, initialize_node) if isinstance(cut_location, 
                    SingleQubitCutLocation) else (cut_czc, cut_czt)

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
                    qubits=[Qubit(cut_location.qubits[0][0], 
                                    cut_location.qubits[0][1])],
                ),
            )

            offset += 1

            circuit_data.insert(
                cut_location.index + offset,
                CircuitInstruction(
                    operation=cur_ops[1],
                    qubits=[Qubit(cut_location.qubits[1][0], 
                                    cut_location.qubits[1][1])],
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
            new.qubits.insert(orig.find_bit(qargs[0]).index+1+offset, q_fresh)
            offset += 1
            # update the mapping so q_old -> q_fresh going forward
            qubit_map[qargs[0]] = q_fresh

        else:
            # just copy the gate over to mapped_qs
            new.append(inst, mapped_qs, cargs)

    return new

def count_gates(qc: QuantumCircuit):
    gate_count = dict.fromkeys(qc.qubits, 0)
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def _remove_idle_wires(qc: QuantumCircuit):
    qc_out = deepcopy(qc)
    gate_count = count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
            for i in qc_out.qregs:
                if qubit in i._bits:
                    i._bits.remove(qubit)
    qc_out.qregs[0]._bit_indices = {
        qubit: qc_out.qubits.index(qubit) for qubit in qc_out.qubits
    }
    qc_out.qregs[0]._bits = qc_out.qubits
    qc_out.qregs[0]._size = len(qc_out.qregs[0]._bits)
    return qc_out


def _separate_subcircuits(circuit):
    dag = circuit_to_dag(circuit)

    circs = dag.separable_circuits()

    new_circs = []
    for i in circs:
        circ = _remove_idle_wires(dag_to_circuit(i))
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
        circ.add_register(ClassicalRegister(circ.num_qubits - clbits_qpd 
                                            + clbits, "meas"))

    return subcircuits


def get_qubit_map(subcircuits: list[QuantumCircuit]):
    def filter_obs_i(qc_data):
        return [i for i in qc_data if "obs" in i.operation.name]

    def sort_func(obs):
        return int(obs.operation.name.split("_")[1])

    map_qubit = {}
    count = 0
    for ind, i in enumerate(reversed(subcircuits)):
        for j in sorted(filter_obs_i(i.data), key=sort_func, reverse=True):
            map_qubit[int(j.operation.name.split("_")[1])] = count
            count += 1

    return map_qubit


def get_locations_and_subcircuits(
    circuit: QuantumCircuit,
    max_qubits: list[int] | None = None,
):
    """Get cut locations and subcircuits with placeholder operations.

    Args:
        circuit (QuantumCircuit): circuit with cuts inserted
        max_qubits (list[int], optional):
            list of maximum qubits per subcircuit when using automatic cut
            finding. If None, no constraint is used. Defaults to None.
            In general it is not necesary to manually specify this parameter.

    Returns:
        tuple: A tuple containing:
            - list[SingleQubitCutLocation]: Locations of the cuts as a list
            - list[QuantumCircuit]: Subcircuits with placeholder operations
            - dict[int:int]: map of subcircuit qubit indices to original circuit
                            qubit indices

    """
    circuit = circuit.copy()  # copy to avoid modifying the original circuit
    for i in range(circuit.num_qubits):
        obs_m = QuantumCircuit(1, name=f"obs_{i}")
        obs_m = obs_m.to_instruction()
        circuit.append(obs_m, [i])
    cut_locations = _get_cut_locations(circuit)
    circuit1 = _insert_cut_nodes(circuit, cut_locations)
    circuit = _move_to_new_wire(circuit1.copy())
    subcircuits = _separate_subcircuits(circuit)
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

    return cut_locations, fixed_circs, map_qubits


def run_cut_circuit(
    subcircuits: list[QuantumCircuit],
    cut_locations: np.ndarray[SingleQubitCutLocation],
    observables: list[int | list[int]],
    map_qubits: dict[int, int],
    backend=AerSimulator(),
) -> np.ndarray[float]:
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
        subcircuits (list[QuantumCircuit]):
            subcircuits containing the placeholder operations
        cut_locations (np.ndarray[CutLocation]): list of cut locations
        observables (list[int | list[int]]):
            list of observables as qubit indices (Z observable)
        backend: backend to use for running experiment circuits (optional)

    Returns:
        list: a list of expectation values

    """

    if not isinstance(backend, AerSimulator):
        transpiled_subcircuits = transpile_subcircuits(subcircuits, 
                                                       cut_locations, backend)
    
        (subexperiments, 
        coefs, 
        id_meas) = get_experiment_circuits(transpiled_subcircuits, 
                                        cut_locations)
    else:
        (subexperiments, 
        coefs, 
        id_meas) = get_experiment_circuits(subcircuits, 
                                        cut_locations)
        
    results = run_experiments(
        subexperiments,
        cut_locations,
        id_meas=id_meas,
        backend=backend,
    )

    return estimate_expectation_values(
        results, coefs, cut_locations, observables, map_qubits
    )


def run(
    circuit: QuantumCircuit,
    observables: list[int, list[int]],
    backend=AerSimulator(),
) -> list[float]:
    """Run the whole circuit knitting sequence with one function call.

    Args:
        circuit (QuantumCircuit): circuit with cut experiments
        observables (list[int | list[int]]):
            list of observbles in the form of qubit indices (Z-obsevable).
        backend: backend to use for running experiment circuits (optional)

    Returns:
        list: a list of expectation values

    """
    # circuit = circuit.copy()
    qss, circs, map_qubits = get_locations_and_subcircuits(circuit)

    return run_cut_circuit(circs, qss, observables, map_qubits, backend)
