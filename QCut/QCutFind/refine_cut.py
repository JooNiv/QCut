def qubits_per_partition(graph, labels):
    """
    Compute the number of unique original qubits present in each partition.

    Approach (robust to undirected edge ordering):
    - Use wire edges (len(data) == 2, data[0] is qubit id) to map nodes -> qubit id.
    - Propagate node->qubit mapping across 2-qubit gate edges using the fact that
      the edge payload contains the pair of qubits; if one endpoint's node is
      known to be q, the other must be the other qubit in the pair.
    - Aggregate per-partition sets of qubits from mapped nodes.
    - Fallback: handle qubits that appear only once (no wire edges). For a gate
      whose both qubits are still unseen, assign both to the single partition if
      both endpoints are in the same partition; if the endpoints are in different
      partitions, assign one qubit to each partition (identity doesn't matter for counts).
    """

    # Collect edges once
    edges = list(graph.edge_index_map().values())
    wire_edges = [e for e in edges if len(e[2]) == 2]
    gate_edges = [e for e in edges if len(e[2]) > 2]

    # 1) Map nodes to original qubit IDs via wire edges
    node_to_qubit = {}
    for u, v, data in wire_edges:
        qubit = data[0]
        # Assign and sanity-check if seen before
        if u in node_to_qubit and node_to_qubit[u] != qubit:
            # Inconsistent graph; keep first seen to avoid oscillations
            pass
        else:
            node_to_qubit[u] = qubit
        if v in node_to_qubit and node_to_qubit[v] != qubit:
            pass
        else:
            node_to_qubit[v] = qubit

    # 2) Propagate across gate edges without relying on endpoint order
    changed = True
    while changed:
        changed = False
        for u, v, data in gate_edges:
            qubits = data[2]
            qset = set(qubits)
            if (
                u in node_to_qubit
                and v not in node_to_qubit
                and node_to_qubit[u] in qset
            ):
                other = qset - {node_to_qubit[u]}
                if len(other) == 1:
                    node_to_qubit[v] = next(iter(other))
                    changed = True
            elif (
                v in node_to_qubit
                and u not in node_to_qubit
                and node_to_qubit[v] in qset
            ):
                other = qset - {node_to_qubit[v]}
                if len(other) == 1:
                    node_to_qubit[u] = next(iter(other))
                    changed = True

    # 3) Aggregate per-partition unique qubits from mapped nodes
    part_qubits = {}
    for node, q in node_to_qubit.items():
        part = labels.get(node)
        if part is None:
            continue
        if part not in part_qubits:
            part_qubits[part] = set()
        part_qubits[part].add(q)

    # 4) Fallback for qubits that appear only once (no wire edges, so not mapped)
    known_qubits = set().union(*part_qubits.values()) if part_qubits else set()
    for u, v, data in gate_edges:
        q0, q1 = data[2]
        # If neither qubit has been seen via wire mapping, attribute by 
        # endpoint partitions
        if q0 not in known_qubits and q1 not in known_qubits:
            parts = set()
            pu = labels.get(u)
            pv = labels.get(v)
            if pu is not None:
                parts.add(pu)
            if pv is not None:
                parts.add(pv)
            if len(parts) == 1:
                p = next(iter(parts))
                part_qubits.setdefault(p, set()).update([q0, q1])
                known_qubits.update([q0, q1])
            elif len(parts) == 2:
                p1, p2 = list(parts)
                part_qubits.setdefault(p1, set()).add(q0)
                part_qubits.setdefault(p2, set()).add(q1)
                known_qubits.update([q0, q1])

    return {p: len(qs) for p, qs in part_qubits.items()}


def extra_wire_cuts(circuit, max_qubits, cut_data_test):
    num_allowed_wire_cuts = circuit.num_qubits - sum(max_qubits)
    num_wire_cuts = len([x for x in cut_data_test if len(x) == 2])
    return num_wire_cuts - num_allowed_wire_cuts


def revert_wire_cut_cost(graph, cut_data_test, cut_data):
    """
    1. get wire cuts from cut_data(_test)
    2. for each wire cut get number of gate_edges on same qubit after cut

    ? How to handle multiple cuts on a wire
        -
    """
    wirecuts = [x for x in zip(cut_data, cut_data_test) if len(x[1]) == 2]

    # for wirecut in wirecuts:
    res = []
    for wirecut in wirecuts:
        leftNode = wirecut[0][0]
        neighbours = graph.neighbors(leftNode)
        filtered = [x for x in neighbours if x > leftNode]
        filtered.sort()
        rightNode = filtered[-1]

        connected_2q_gates = []
        nodes_to_explore = [rightNode]
        prev_leftNodes = [leftNode]

        for i in nodes_to_explore:
            neighbours = graph.neighbors(i)
            print("to explore: ", nodes_to_explore)
            print("connected: ", connected_2q_gates)
            print("prev: ", prev_leftNodes)
            print("node: ", i)
            for n in neighbours:
                if n in prev_leftNodes:
                    continue
                edge = graph.get_edge_data(i, n)
                if len(edge) == 2:
                    nodes_to_explore.append(n)
                    prev_leftNodes.append(i)
                else:
                    connected_2q_gates.append((i, n, edge))

            print(nodes_to_explore)
            print(connected_2q_gates)

        res.append(
            {
                "cost": len(connected_2q_gates),
                "data": wirecut,
                "connected": connected_2q_gates,
            }
        )

    return res


def revert_wire_cuts(wirecuts, cut_data_test, cut_data, labels, nodes_on_qubit, num):
    for wirecut in wirecuts[:num]:
        cut_data_test.remove(wirecut["data"][1])
        cut_data.remove(wirecut["data"][0])
        to_flip = {
            
                x
                for x in nodes_on_qubit[wirecut["data"][1][0]]
                if x > wirecut["data"][0][0]
            
        }
        for x in wirecut["connected"]:
            cut_data_test.append(x[2])
            cut_data.append(x)

        print(labels[wirecut["data"][0][0]])
        print(to_flip)
        for i in to_flip:
            print(labels[i])
            labels[i] = labels[wirecut["data"][0][0]]


def give_receive_qubits(qubits_per_partition, max_qubits):
    copy_max_qubits = max_qubits.copy()
    res = {}
    for key, value in qubits_per_partition.items():
        closest = min(
            [(x, x - value, abs(x - value)) for x in copy_max_qubits],
            key=lambda y: y[2],
        )

        res[key] = {"receive": closest[1], "q": value, "max_q": closest[0]}
        copy_max_qubits.remove(closest[0])

    return res


def swap_qubits(
    graph, cut_data_test, cut_data, receivers, givers, labels, nodes_on_qubit
):
    flat_list = [item for x in cut_data for item in [x[0], x[1]] if len(x[2]) > 2]
    print(flat_list)
    for elem in givers:
        label = list(elem.keys())[0]
        for i in range(elem[label]):
            print(
                f"Giver {label} has {elem[label]} extra qubits, swapping {i + 1} times"
            )
            valid = {}
            for qubit_label, nodes in nodes_on_qubit.items():
                # not sure if this will hold
                valid_nodes = list(nodes)
                if valid_nodes:
                    cost = len([x for x in valid_nodes if x not in flat_list])
                    valid[qubit_label] = (valid_nodes, cost)
            validlist = list(valid.items())
            validlist.sort(key=lambda x: x[1][1])
            print(f"valid: {validlist}")

            for i in validlist[: i + 1]:
                print(f"Swapping qubit {i[0]}")

                # Collect all to_cut gates for all nodes first
                all_to_cut = []
                for node in i[1][0]:
                    to_cut = [x for x in graph.in_edges(node) if len(x[2]) > 2]
                    all_to_cut.extend(to_cut)
                # Calculate receiver label once from all gates
                receiver_label = None
                for gate in all_to_cut:
                    if labels[gate[0]] != labels[gate[1]]:
                        # Choose the label that is not equal to current label
                        receiver_label = (
                            labels[gate[1]]
                            if labels[gate[0]] == label
                            else labels[gate[0]]
                        )
                        break

                if receiver_label is None and receivers:
                    # Use the first receiver in the list
                    receiver_entry = receivers[0]
                    receiver_label = list(receiver_entry.keys())[0]
                    receiver_entry[receiver_label] -= 1
                    if receiver_entry[receiver_label] == 0:
                        receivers.pop(0)

                print(f"Receiver label: {receiver_label}")

                # Apply the same receiver label to all gates
                for gate in all_to_cut:
                    print(labels[gate[0]], labels[gate[1]])
                    print(f"Cutting {gate}")
                    if gate in cut_data:
                        cut_data.remove(gate)
                        cut_data_test.remove(gate[2])

                    else:
                        cut_data_test.append(gate[2])
                        cut_data.append(gate)

                    labels[gate[0]] = receiver_label
                    labels[gate[1]] = receiver_label


def refine_cuts(
    circuit,
    nodes_on_qubit,
    cut_data,
    cut_data_test,
    labels,
    graph,
    max_qubits,
    verbose=True,
):
    cut_data = cut_data.copy()
    cut_data_test = cut_data_test.copy()
    labels = labels.copy()

    extra_wire_cuts_val = extra_wire_cuts(circuit, max_qubits, cut_data_test)

    if extra_wire_cuts_val > 0:
        if verbose:
            print(f"Extra wire cuts: {extra_wire_cuts_val}")
            print("Cost:")
        cost = revert_wire_cut_cost(graph, cut_data_test, cut_data)
        cost.sort(key=lambda x: x["cost"])
        if verbose:
            print(cost[:extra_wire_cuts_val])
            print("Reverting wire cuts...")
        revert_wire_cuts(
            cost, cut_data_test, cut_data, labels, nodes_on_qubit, extra_wire_cuts_val
        )

    if verbose:
        print("Calculating qubits per partition...")
    qubits_per_partition_val = qubits_per_partition(graph, labels)
    if verbose:
        print("Qubits per partition:", qubits_per_partition_val)
        print("Calculating give/receive qubits...")
    res = give_receive_qubits(qubits_per_partition_val, max_qubits)
    receivers = [
        {key: value["receive"]} for key, value in res.items() if value["receive"] > 0
    ]
    givers = [
        {key: abs(value["receive"])}
        for key, value in res.items()
        if value["receive"] < 0
    ]
    if verbose:
        print("Receivers:", receivers)
        print("Givers:", givers)
        print("Swapping qubits...")
    swap_qubits(
        graph, cut_data_test, cut_data, receivers, givers, labels, nodes_on_qubit
    )

    cut_data_test.sort(key=lambda x: x[0])
    cut_data.sort(key=lambda x: x[2][0])

    return cut_data, cut_data_test, labels
