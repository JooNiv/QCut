"""
The main cut finding workflow for QCut.
"""

import rustworkx as rx
from qiskit import transpile
from qiskit.circuit import CircuitInstruction

from QCut.circuit_preparation import get_locations_and_subcircuits
from QCut.consolidate import consolidate_two_qubit_blocks
from QCut.options import CutOptions, resolve
from QCut.QCutFind.graph_circuit_utils import circ_to_graph
from QCut.QCutFind.metis import k_way_metis_partition
from QCut.QCutFind.refine import refine_cuts
from QCut.qpd_gates import QPD_GATE_REGISTRY, CutTwoQubitGate
from QCut.qpd_gates import cut_op as cut

#: Basis the circuit is transpiled into before the interaction graph is built. The
#: two-qubit entries stop transpilation from breaking apart gates that could be cut
#: whole, which would cost one cut per cz rather than one cut at the gate's own gamma.
BASIS_GATES = [
    "r",
    "u",
    "cz",
    "swap",
    "iswap",
    "cx",
    "cy",
    "ch",
    "ecr",
    "dcx",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "crx",
    "cry",
    "crz",
    "cp",
    "xx_plus_yy",
    "xx_minus_yy",
]


def extract_cuts(graph, labels):  # noqa: C901
    cut_edges = []

    for u, v in graph.edge_list():
        if labels[u] != labels[v]:
            cut_edges.append((u, v))

    cut_data_test = []

    for edge in cut_edges:
        cut_data_test.append(graph.get_edge_data(edge[0], edge[1]))
        # Collect all unique nodes involved in cut_edges
    cut_nodes = set()
    cut_data = []
    for u, v in cut_edges:
        # Get all edges connected to u and v (excluding the cut edge itself)
        if len(graph.get_edge_data(u, v)) > 2:
            cut_data.append((u, v, graph.get_edge_data(u, v)))
            continue
        connected_edges = []
        for nbr in graph.neighbors(u):
            if nbr != v:
                data = graph.get_edge_data(u, nbr)
                if len(data) > 2:
                    connected_edges.append((u, nbr, data))
        for nbr in graph.neighbors(v):
            if nbr != u:
                data = graph.get_edge_data(v, nbr)
                if len(data) > 2:
                    connected_edges.append((v, nbr, data))
        # Find the edge with the smallest data[1]
        if len(connected_edges) > 0:
            filtered_edges = [
                edge
                for edge in connected_edges
                if graph.get_edge_data(u, v)[0] in edge[2][2]
            ]
            min_edge = min(filtered_edges, key=lambda x: x[2])
            cut_data.append(min_edge)

        cut_nodes.add(u)
        cut_nodes.add(v)

    return cut_data, cut_data_test


def insert_or_append(circuit, index, instruction):
    """
    Insert an instruction at a specific index in the circuit.
    If the index is out of bounds, append the instruction.
    """
    if index < 0 or index >= len(circuit.data):
        circuit.data.append(instruction)
    else:
        circuit.data.insert(index, instruction)


def add_cuts_to_circuit(circuit, cut_data, cut_data_test):
    qctest = circuit.copy()
    offset = 0

    zipped_data = zip(cut_data, cut_data_test)
    zipped_data = sorted(
        zipped_data,
        key=lambda pair: (
            pair[0][2][0],
            -len(pair[1]) if hasattr(pair[1], "__len__") else 0,
        ),
    )

    for ind, (i, j) in enumerate(zipped_data):
        if len(j) > 2:
            # Remove the original operation at the specified index§
            qubits = list(
                filter(
                    lambda x: x is not None,
                    [
                        q if circuit.find_bit(q).index in j[2] else None
                        for q in qctest.qubits
                    ],
                )
            )
            target_index = i[2][0] + offset
            original = qctest.data.pop(target_index)

            gate_name = j[1]
            if gate_name in QPD_GATE_REGISTRY:
                cut_marker = QPD_GATE_REGISTRY[gate_name]
            else:
                # No dedicated marker, so wrap the gate we just removed and let its QPD
                # be generated from its matrix at expansion time.
                cut_marker = CutTwoQubitGate(original.operation)

            # Insert the cut operation
            insert_or_append(
                qctest,
                target_index,
                CircuitInstruction(cut_marker, qubits),
            )

        else:
            # Find the correct index for the cut operation
            ind = i[2][2][i[2][2].index(j[0])]
            target_index = i[2][0] + offset + 1

            # Insert the cut operation
            insert_or_append(
                qctest,
                target_index,
                CircuitInstruction(cut, [qctest.qubits[ind]]),
            )

            # Increment offset to account for the new instruction
            offset += 1

    return qctest


def find_cuts(  # noqa: C901
    circuit,
    num_partitions: int | None = None,
    max_qubits=None,
    cuts="both",
    more_data=False,
    options: CutOptions | None = None,
):
    """Partition a quantum circuit into subcircuits by inserting cut operations.

    Converts the input circuit into a graph representation and partitions it into the
    specified number of subcircuits using METIS partitioning. Optionally refines the
    partitioning to respect maximum qubit constraints per subcircuit. Cut operations
    are inserted at the partition boundaries, and the resulting subcircuits and their
    mappings are returned.

    Args:
        circuit (QuantumCircuit): The input quantum circuit to partition.
        num_partitions (int, optional): Number of partitions to create. If None,
            determined from max_qubits.
        max_qubits (list[int], optional): Maximum number of qubits allowed in each
            partition. If specified, must match num_partitions.
        cuts (str, optional): Type of cuts to insert. Can be "both", "gate", or "wire".
            Defaults to "both".
        more_data (bool, optional): If True, returns additional data for debugging and
            analysis. Defaults to False.
        options (CutOptions, optional): configuration for the run. Defaults to
            QCut.options.DEFAULT_OPTIONS.

    Returns:
        tuple: If more_data is False, returns:
            - list[list[int]]: Qubit locations for each subcircuit.
            - list[QuantumCircuit]: List of subcircuits after cuts.
            - list[dict]: Mapping of qubits for each subcircuit.
        If more_data is True, returns:
            - list[list[int]]: Qubit locations for each subcircuit.
            - list[QuantumCircuit]: List of subcircuits after cuts.
            - list[dict]: Mapping of qubits for each subcircuit.
            - QuantumCircuit: The circuit with cuts inserted.
            - list: Data describing the cuts.
            - list: Additional cut data for testing.
            - dict: Partition labels for each node.
            - rustworkx.PyGraph: The graph representation of the circuit.
            - dict: Mapping of nodes to qubits.
    """

    if num_partitions is None:
        if max_qubits is None:
            raise ValueError("Either num_partitions or max_qubits must be specified.")
        num_partitions = len(max_qubits)
    elif max_qubits is not None and len(max_qubits) != num_partitions:
        raise ValueError(
            "If both num_partitions and max_qubits are specified, length of"
            "max_qubits must match num_partitions."
        )

    if (max_qubits is not None and len(max_qubits) < 2) or num_partitions < 2:
        raise ValueError("Number of partitions has to be atleast 2")

    if num_partitions == 1:
        return circuit, [], []

    circuit.remove_final_measurements()

    options = resolve(options)
    circuit = transpile(circuit, optimization_level=0, basis_gates=BASIS_GATES)
    if options.consolidate:
        # Before the graph is built, so a merged run shows up as one edge to cut rather
        # than several. Nothing is marked yet, so every pair is a candidate.
        circuit = consolidate_two_qubit_blocks(circuit)

    graph, nodes_on_qubit = circ_to_graph(circuit, mode=cuts)

    components = rx.connected_components(graph)
    if len(components) == num_partitions:
        labels = {}
        for comp_ind, comp in enumerate(components):
            for node in comp:
                labels[node] = comp_ind
        cut_data, cut_data_test = [], []
    else:
        labels = k_way_metis_partition(graph, num_partitions)
        cut_data, cut_data_test = extract_cuts(graph, labels)

    if max_qubits is not None:
        cut_data, cut_data_test, labels = refine_cuts(
            cut_data,
            cut_data_test,
            labels,
            graph,
            max_qubits,
            nodes_on_qubit,
            circuit,
            cuts,
        )

    cut_circuit = add_cuts_to_circuit(circuit, cut_data, cut_data_test)

    if max_qubits is not None:
        final_cut_circuit = get_locations_and_subcircuits(
            cut_circuit, max_qubits=max_qubits, options=options
        )
    else:
        final_cut_circuit = get_locations_and_subcircuits(cut_circuit, options=options)

    if not more_data:
        return final_cut_circuit
    else:
        return (
            final_cut_circuit,
            cut_circuit,
            cut_data,
            cut_data_test,
            labels,
            graph,
            nodes_on_qubit,
        )
