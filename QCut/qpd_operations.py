"""
A module for generating the QPD operations and inserting them into the circuit at
the appropriate locations.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from qiskit.circuit import (
    CircuitInstruction,
    Qubit,
)

from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.qpd import cz_qpd, identity_qpd


def _insert_wire_cut_qpd(
    ind,
    op,
    subcircuit,
    offset,
    qpd_qubits,
    qpd,
    classical_bit_index,
    inserted_operations,
):
    if "Meas" in op.operation.name:  # if measure channel remove placeholder
        # and insert current
        # qpd operation
        qubit_index = subcircuit.find_bit(op.qubits[0]).index
        subcircuit.data.pop(ind + offset)  # remove plaxceholder
        # measure channel
        qpd_qubits.append(qubit_index)  # store index
        qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
        meas_op = qpd[int(op.operation.name.split("_")[-1])]["op_0"]
        if meas_op.name == "id-meas":  # if identity measure channel
            # store indices
            # remove extra classical bits and registers
            # _adjust_cregs(subcircuit)
            for subop in reversed(meas_op.data):
                subcircuit.data.insert(
                    ind + offset,
                    CircuitInstruction(
                        operation=subop.operation, qubits=qubits_for_operation
                    ),
                )
        else:
            for i, subop in enumerate(reversed(meas_op.data)):
                if i == 0:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                            clbits=[subcircuit.cregs[0][classical_bit_index]],
                        ),
                    )
                else:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                        ),
                    )

            # increment classical bit counter
            classical_bit_index += 1

        inserted_operations += 1
        offset += len(meas_op.data) - 1

    if "Init" in op.operation.name:
        subcircuit.data.pop(ind + offset)
        init_op = qpd[int(op.operation.name.split("_")[-1])]["op_1"]
        qubits_for_operation = [
            Qubit(subcircuit.qregs[0], subcircuit.find_bit(x).index) for x in op.qubits
        ]
        for subop in reversed(init_op.data):
            subcircuit.data.insert(
                ind + offset,
                CircuitInstruction(
                    operation=subop.operation, qubits=qubits_for_operation
                ),
            )

        inserted_operations += 1
        offset += len(init_op) - 1

    return offset, classical_bit_index, inserted_operations


def _insert_cz_cut_qpd(  # noqa: C901
    ind,
    op,
    subcircuit,
    offset,
    qpd_qubits,
    qpd,
    classical_bit_index,
    inserted_operations,
):
    if "_c_" in op.operation.name:  # if measure channel remove placeholder
        # and insert current
        # qpd operation

        qubit_index = subcircuit.find_bit(op.qubits[0]).index
        subcircuit.data.pop(ind + offset)  # remove plaxceholder
        # measure channel
        # qpd_qubits.append(qubit_index)  # store index
        qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
        meas_op = qpd[int(op.operation.name.split("_")[-1])]["op_0"]
        if meas_op.name in ["id-meas", "s", "sdg", "z"]:
            # if identity measure channel
            # store indices
            # remove extra classical bits and registers
            # if meas_op.name != "id-meas":
            #    _adjust_cregs(subcircuit)
            for subop in reversed(meas_op.data):
                subcircuit.data.insert(
                    ind + offset,
                    CircuitInstruction(
                        operation=subop.operation, qubits=qubits_for_operation
                    ),
                )
        else:
            for i, subop in enumerate(reversed(meas_op.data)):
                if i == 0:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                            clbits=[subcircuit.cregs[0][classical_bit_index]],
                        ),
                    )
                else:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                        ),
                    )

            # increment classical bit counter
            classical_bit_index += 1

        inserted_operations += 1
        offset += len(meas_op) - 1

    if "_t_" in op.operation.name:
        # and insert current
        # qpd operation
        qubit_index = subcircuit.find_bit(op.qubits[0]).index
        subcircuit.data.pop(ind + offset)  # remove plaxceholder
        # measure channel
        # qpd_qubits.append(qubit_index)  # store index
        qubits_for_operation = [Qubit(subcircuit.qregs[0], qubit_index)]
        meas_op = qpd[int(op.operation.name.split("_")[-1])]["op_1"]
        if meas_op.name in ["id-meas", "s", "sdg", "z"]:
            # if identity measure channel
            # store indices

            # remove extra classical bits and registers
            # if meas_op.name != "id-meas":
            #    _adjust_cregs(subcircuit)
            for subop in reversed(meas_op.data):
                subcircuit.data.insert(
                    ind + offset,
                    CircuitInstruction(
                        operation=subop.operation, qubits=qubits_for_operation
                    ),
                )
        else:
            for i, subop in enumerate(reversed(meas_op.data)):
                if i == 0:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                            clbits=[subcircuit.cregs[0][classical_bit_index]],
                        ),
                    )
                else:
                    subcircuit.data.insert(
                        ind + offset,
                        CircuitInstruction(
                            operation=subop.operation,
                            qubits=qubits_for_operation,
                        ),
                    )

            # increment classical bit counter
            classical_bit_index += 1

        inserted_operations += 1
        offset += len(meas_op) - 1

    return offset, classical_bit_index, inserted_operations


def get_qpd_combinations(
    cut_locations: list[CutLocation | SingleQubitCutLocation],
) -> Iterable[tuple[dict]]:
    """Get all possible combinations of the QPD operations so that each combination
    has len(cut_locations) elements.

    Args:
        cut_locations (list[CutLocation | SingleQubitCutLocation]): cut locations

    Returns:
        Iterable[tuple[dict]]:
            Iterable of the possible QPD operations

    """
    qpd_lists = []
    for cut in cut_locations:
        if isinstance(cut, SingleQubitCutLocation):
            qpd_lists.append(identity_qpd)
        elif isinstance(cut, CutLocation):
            qpd_lists.append(cz_qpd)
        else:
            raise TypeError(f"Unknown cut type: {type(cut)}")

    # Cartesian product across all qpd options per cut
    all_combinations = product(*qpd_lists)
    return all_combinations
