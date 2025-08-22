import rustworkx as rx
from QCutFind.graph_circuit_utils import circ_to_graph
from QCutFind.ilp import k_way_min_cut_cp_sat
from QCutFind.kmeans import k_way_kmeans_partition
from QCutFind.metis import k_way_metis_partition
from QCutFind.refine import refine_cuts
from QCutFind.spectral import (
    k_way_spectral_partition,
    k_way_spectral_partition_sk,
)
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

from QCut import cut


def extract_cuts(graph, labels):
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

    gate_cut_ind = 0
    for ind, (i, j) in enumerate(zipped_data):
        cutCZc = QuantumCircuit(1, name=f"cutCZc_{gate_cut_ind}").to_instruction()
        cutCZt = QuantumCircuit(1, name=f"cutCZt_{gate_cut_ind}").to_instruction()

        if len(j) > 2:
            # Remove the original operation at the specified index
            target_index = i[2][0] + offset
            qctest.data.pop(target_index)

            # Insert the first cut operation
            insert_or_append(
                qctest,
                target_index,
                CircuitInstruction(cutCZc, [qctest.qubits[j[2][0]]]),
            )

            # Insert the second cut operation
            insert_or_append(
                qctest,
                target_index + 1,  # Adjust for the first insertion
                CircuitInstruction(cutCZt, [qctest.qubits[j[2][1]]]),
            )

            # Increment offset to account for the two new instructions
            offset += 1  # Only increment by 1 since one operation was removed and
            # two were added
            gate_cut_ind += 1

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


def find_cuts(
    circuit,
    num_partitions=None,
    max_qubits=None,
    cuts="both",
    method="spectral",
):
    """
    Find cuts in a quantum circuit and return a new circuit with cut operations 
    inserted.

    Args:
        circuit (QuantumCircuit): The input quantum circuit.
        gate_cuts (bool): If True, use gate cuts and wire cuts; otherwise, use only 
        wire cuts.

    Returns:
        QuantumCircuit: A new quantum circuit with cut operations inserted.
    """

    if num_partitions is None and max_qubits is not None:
        num_partitions = len(max_qubits)
    elif num_partitions is None and max_qubits is None:
        raise ValueError("Either num_partitions or max_qubits must be specified.")
    elif num_partitions is not None and max_qubits is not None:
        if len(max_qubits) != num_partitions:
            raise ValueError(
                "If both num_partitions and max_qubits are specified, length of" \
                "max_qubits must match num_partitions."
            )

    if num_partitions < 1:
        raise ValueError(
            "max_qubits_per_circuit must be less than the number of qubits in the" \
            "circuit."
        )
    if num_partitions == 1:
        return circuit, [], []

    gate_cut_weight = 3 if (cuts == "both" or cuts == "gate") else 1000000000
    wire_cut_weight = 4 if (cuts == "both" or cuts == "wire") else 1000000000
    import time

    start = time.time()
    graph, nodes_on_qubit = circ_to_graph(
        circuit, gateCutWeight=gate_cut_weight, wireCutWeight=wire_cut_weight
    )

    components = rx.connected_components(graph)
    print(f"Components found {len(components)}: ", components)
    labels = {}
    if len(components) == num_partitions:
        for comp_ind, comp in enumerate(components):
            for node in comp:
                labels[node] = comp_ind
        return circuit, [], [], labels, graph, nodes_on_qubit

    print("Circ to graph took: ", time.time() - start)

    print(len(graph.nodes()), "nodes in graph")
    print(len(graph.edges()), "edges in graph")

    start = time.time()
    if method == "metis":
        labels = k_way_metis_partition(graph, num_partitions)
    elif method == "spectral":
        labels = k_way_spectral_partition(graph, num_partitions)
    elif method == "spectral_sk":
        labels = k_way_spectral_partition_sk(graph, num_partitions)
    elif method == "kmeans":
        labels = k_way_kmeans_partition(graph, num_partitions)
    elif method == "ilp":
        labels = k_way_min_cut_cp_sat(graph, num_partitions)
    else:
        raise ValueError(
            "Invalid method. Existing methods: 'metis', 'spectral', 'spectral_sk'," \
            "'kmeans', or 'ilp'."
        )

    print("Partitioning took: ", time.time() - start)

    start = time.time()
    cut_data, cut_data_test = extract_cuts(graph, labels)
    print("Extracting took: ", time.time() - start)

    print("Initial labels: ", labels)

    if max_qubits is not None:
        print("Refining:")
        start = time.time()
        cut_data, cut_data_test, labels = refine_cuts(
            cut_data,
            cut_data_test,
            labels,
            graph,
            max_qubits,
            nodes_on_qubit,
            circuit,
            onlywire=False,
        )
        print("Refining took: ", time.time() - start)
    else:
        print("No refinement needed, max_qubits is None.")

    """per_partition = qubits_per_partition(graph, labels)

    start = time.time()
    cut_data, cut_data_test = extract_cuts(graph, labels)
    print("Extracting took: ", time.time() - start)


    if max_qubits is not None:
        start = time.time()
        res = give_receive_qubits(per_partition, max_qubits)
        givers = [{key: abs(value["receive"])}
                for key, value in res.items() if value["receive"] < 0]
        
        if len(givers) > 0:
            cut_data, cut_data_test, labels = refine_cuts(circuit, nodes_on_qubit, 
            cut_data, cut_data_test, labels, graph, max_qubits, verbose=True)
        print("Refining cuts took: ", time.time() - start)
        print("Refined labels: ", labels)"""

    start = time.time()
    cut_circuit = add_cuts_to_circuit(circuit, cut_data, cut_data_test)
    print("Adding cuts took: ", time.time() - start)

    return cut_circuit, cut_data, cut_data_test, labels, graph, nodes_on_qubit
