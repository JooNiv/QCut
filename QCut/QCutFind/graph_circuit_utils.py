import rustworkx as rx


def weight_fn(edge_data):
    """
    Custom weight function for edges.
    """
    if len(edge_data) > 2:
        # your existing logic: tuple[3] is raw_weight
        return edge_data[3]  # Use the weight from the tuple
    # default weight
    return edge_data[1]


max_added = None


def update_added(added_set, added, op):
    """
    Append the two new node indices for a 2-qubit op,
    using a global next-available counter instead of
    scanning `added`.
    """
    global max_added

    q0, q1 = op[1][2]  # the two qubit indices

    # lazy initialize on first call
    if max_added is None:
        if added:
            max_added = max(added)
        else:
            # for the very first gate, we'll set it below
            max_added = None

    # both qubits have been seen before → allocate two brand-new nodes
    if q0 in added_set and q1 in added_set:
        a = max_added + 1
        b = max_added + 2
        added.extend([a, b])
        added_set.add(a)
        added_set.add(b)
        max_added += 2

    # only q0 seen → reuse q1 and allocate one fresh node
    elif q0 in added_set:
        a = q1
        max_added = max(max_added, a)
        b = max_added + 1
        added.extend([a, b])
        added_set.add(b)
        added_set.add(a)
        max_added += 1

    # only q1 seen → reuse q0 and allocate one fresh node
    elif q1 in added_set:
        a = q0
        max_added = max(max_added, a)
        b = max_added + 1
        added.extend([a, b])
        added_set.add(b)
        added_set.add(a)
        max_added += 1

    # neither seen → just append those original qubit-IDs
    else:
        added.extend([q0, q1])
        added_set.add(q0)
        added_set.add(q1)
        # now that they’re in, prime max_added for future
        max_added = max(q0, q1)


def update_nodes_on_qubit(nodes_on_qubit, qubit, node):
    if qubit not in nodes_on_qubit:
        nodes_on_qubit[qubit] = [node]
    else:
        nodes_on_qubit[qubit].append(node)


def circ_to_graph(circuit, gateCutWeight=1000000000, wireCutWeight=4):  # noqa: C901
    """
    Convert a quantum circuit to a graph representation.

    Args:
        circuit: A quantum circuit object.

    Returns:
        A graph representation of the circuit.
    """

    nodes_on_qubit = {}

    ops_2qb = [
        (
            idx,
            (
                x.operation.name,
                x.operation.num_qubits,
                [circuit.find_bit(i).index for i in x.qubits],
            ),
        )
        for idx, x in enumerate(circuit.data)
        if x.operation.num_qubits == 2
    ]

    G = rx.PyGraph()
    indxs = list(
        range(
            max(
                max([x[0] for x in ops_2qb]),
                len(ops_2qb) * 2,
            )
            + 1
        )
    )
    for i in indxs:
        G.add_node(i)

    track_nodes = {}
    added = []
    added_set = set()

    for ind, op in enumerate(ops_2qb):
        if (
            op[1][2][0] not in track_nodes.keys()
            and op[1][2][1] not in track_nodes.keys()
        ):
            update_added(added_set, added, op)
            if op[1][2][0] in added_set:
                track_nodes[op[1][2][0]] = [(added[-2])]
            else:
                track_nodes[op[1][2][0]] = []
            if op[1][2][1] in added_set:
                track_nodes[op[1][2][1]] = [added[-1]]
            else:
                track_nodes[op[1][2][1]] = []

            update_nodes_on_qubit(nodes_on_qubit, op[1][2][0], added[-2])
            update_nodes_on_qubit(nodes_on_qubit, op[1][2][1], added[-1])

            G.add_edge(
                added[-2],
                added[-1],
                (op[0], op[1][0], op[1][2], gateCutWeight),
            )
        elif op[1][2][1] not in track_nodes.keys():
            update_added(added_set, added, op)

            if op[1][2][1] in added_set:
                track_nodes[op[1][2][1]] = [(added[-1])]
            else:
                track_nodes[op[1][2][1]] = []

            track_nodes[op[1][2][0]].append(added[-2])
            ind = track_nodes[op[1][2][0]]

            update_nodes_on_qubit(nodes_on_qubit, op[1][2][0], added[-2])
            update_nodes_on_qubit(nodes_on_qubit, op[1][2][1], added[-1])

            G.add_edge(
                added[-2],
                added[-1],
                (op[0], op[1][0], op[1][2], gateCutWeight),
            )
            if ind == []:
                G.add_edge(op[1][2][0], added[-1], (op[1][2][0], wireCutWeight))
            else:
                if len(ind) == 1:
                    G.add_edge(ind[-1], op[1][2][0], (op[1][2][0], wireCutWeight))
                else:
                    G.add_edge(ind[-1], ind[-2], (op[1][2][0], wireCutWeight))
        elif op[1][2][0] not in track_nodes.keys():
            update_added(added_set, added, op)

            if op[1][2][0] in added_set:
                track_nodes[op[1][2][0]] = [(added[-2])]
            else:
                track_nodes[op[1][2][0]] = []

            track_nodes[op[1][2][1]].append(added[-1])
            ind = track_nodes[op[1][2][1]]

            update_nodes_on_qubit(nodes_on_qubit, op[1][2][0], added[-2])
            update_nodes_on_qubit(nodes_on_qubit, op[1][2][1], added[-1])

            G.add_edge(
                added[-2],
                added[-1],
                (op[0], op[1][0], op[1][2], gateCutWeight),
            )
            if ind == []:
                G.add_edge(op[1][2][1], added[-1], (op[1][2][1], wireCutWeight))
            else:
                if len(ind) == 1:
                    G.add_edge(ind[-1], op[1][2][1], (op[1][2][1], wireCutWeight))
                else:
                    G.add_edge(ind[-1], ind[-2], (op[1][2][1], wireCutWeight))
        else:
            update_added(added_set, added, op)
            track_nodes[op[1][2][0]].append(added[-2])
            track_nodes[op[1][2][1]].append(added[-1])
            ind0 = track_nodes[op[1][2][0]]
            ind1 = track_nodes[op[1][2][1]]

            update_nodes_on_qubit(nodes_on_qubit, op[1][2][0], added[-2])
            update_nodes_on_qubit(nodes_on_qubit, op[1][2][1], added[-1])

            G.add_edge(
                added[-2],
                added[-1],
                (op[0], op[1][0], op[1][2], gateCutWeight),
            )
            if ind0 == []:
                G.add_edge(op[1][2][0], added[-1], (op[1][2][0], wireCutWeight))
            else:
                if len(ind0) == 1:
                    G.add_edge(ind0[-1], op[1][2][0], (op[1][2][0], wireCutWeight))
                else:
                    G.add_edge(ind0[-1], ind0[-2], (op[1][2][0], wireCutWeight))
            if ind1 == []:
                G.add_edge(op[1][2][1], added[-2], (op[1][2][1], wireCutWeight))
            else:
                if len(ind1) == 1:
                    G.add_edge(ind1[-1], op[1][2][1], (op[1][2][1], wireCutWeight))
                else:
                    G.add_edge(ind1[-1], ind1[-2], (op[1][2][1], wireCutWeight))

    # Remove nodes with degree 0 (not connected to anything)
    isolated_nodes = [node for node in G.node_indices() if G.degree(node) == 0]

    G.remove_nodes_from(isolated_nodes)
    G, nodes_on_qubit = final_graph(G, nodes_on_qubit)
    return G, nodes_on_qubit


def final_graph(graph, nodes_on_qubit):
    # Relabel nodes to ensure consecutive numbering
    # Since rustworkx doesn't have relabel_nodes, we need to create a new graph manually
    old_to_new = {
        old_idx: new_idx for new_idx, old_idx in enumerate(graph.node_indices())
    }

    # Create a new graph
    new_graph = rx.PyGraph()

    # Add nodes with consecutive indices (0, 1, 2, ...)
    for i in range(len(graph.node_indices())):
        new_graph.add_node(i)

    # Add edges with remapped indices
    for edge in graph.edge_list():
        old_source, old_target = edge
        new_source = old_to_new[old_source]
        new_target = old_to_new[old_target]
        edge_data = graph.get_edge_data(old_source, old_target)
        new_graph.add_edge(new_source, new_target, edge_data)

    # Update nodes_on_qubit dictionary with new node indices
    updated_nodes_on_qubit = {}
    for qubit, old_node_list in nodes_on_qubit.items():
        updated_nodes_on_qubit[qubit] = [
            old_to_new[old_node] for old_node in old_node_list if old_node in old_to_new
        ]

    return new_graph, updated_nodes_on_qubit
