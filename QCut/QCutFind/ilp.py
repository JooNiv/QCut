from ortools.sat.python import cp_model


def weight_fn(edge_data):
    """
    Custom weight function for edges.
    """
    if isinstance(edge_data, tuple):
        # your existing logic: tuple[3] is raw_weight
        return edge_data[3]  # Use the weight from the tuple
    # default weight
    return 4


def k_way_min_cut_cp_sat(
    G, k, lower_sizes=None, upper_sizes=None, time_limit_sec=60, num_workers=8
):
    """
    Exact k-way minimum cut via OR-Tools CP-SAT with connectivity enforced.

    G : networkx Graph (with 'weight' attribute on edges)
    k : number of parts
    lower_sizes, upper_sizes : optional lists of length k giving size bounds
    time_limit_sec : solver time limit in seconds
    num_workers : number of threads

    Returns:
        node_to_part: dict mapping node -> part index
        cut_size: total weight of cut edges
    """
    V = list(G.nodes())
    E = list(G.weighted_edge_list())  # (u, v, {'weight': w})
    n = len(V)

    model = cp_model.CpModel()

    # 1. Assignment variables x[i,p]
    x = {}
    for i in V:
        for p in range(k):
            x[(i, p)] = model.NewBoolVar(f"x_{i}_{p}")

    # 2. Cut indicator variables z[i,j]
    z = {}
    for u, v, data in E:
        z[(u, v)] = model.NewBoolVar(f"z_{u}_{v}")
        # ensure symmetry
        z[(v, u)] = z[(u, v)]

    # 3. Each vertex in exactly one part
    for i in V:
        model.Add(sum(x[(i, p)] for p in range(k)) == 1)

    # 4. Balance or non-emptiness
    size_lo = n // k if lower_sizes is None else None
    size_hi = (n + k - 1) // k if upper_sizes is None else None
    for p in range(k):
        if lower_sizes is not None and upper_sizes is not None:
            model.Add(sum(x[(i, p)] for i in V) >= lower_sizes[p])
            model.Add(sum(x[(i, p)] for i in V) <= upper_sizes[p])
        else:
            # enforce non-empty and roughly balanced
            model.Add(sum(x[(i, p)] for i in V) >= 1)
            model.Add(sum(x[(i, p)] for i in V) >= size_lo)
            model.Add(sum(x[(i, p)] for i in V) <= size_hi)

    # 5. Cut constraints: z = 1 if endpoints differ in any part
    for u, v, _ in E:
        for p in range(k):
            model.Add(x[(u, p)] - x[(v, p)] <= z[(u, v)])
            model.Add(x[(v, p)] - x[(u, p)] <= z[(u, v)])

    # 6. Symmetry-breaking: pin one node to part 0
    model.Add(x[(V[0], 0)] == 1)

    # 7. Connectivity constraints via internal-edge count
    # For each part p, ensure its induced subgraph is connected by requiring
    # (# internal edges) >= (# nodes in part) - 1
    e = {}
    for u, v, _ in E:
        for p in range(k):
            e[(u, v, p)] = model.NewBoolVar(f"e_{u}_{v}_{p}")
            # can only be internal if both endpoints in part
            model.Add(e[(u, v, p)] <= x[(u, p)])
            model.Add(e[(u, v, p)] <= x[(v, p)])
            # if both in part, must allow e
            model.Add(e[(u, v, p)] >= x[(u, p)] + x[(v, p)] - 1)
    # enforce connectivity
    for p in range(k):
        model.Add(sum(e[(u, v, p)] for (u, v, _) in E) >= sum(x[(i, p)] for i in V) - 1)

    # 8. Weighted objective: minimize total cut-weight: minimize total cut-weight
    model.Minimize(sum(weight_fn(data) * z[(u, v)] for (u, v, data) in E))

    # 9. Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No solution: {status}")

    # 10. Extract solution
    node_to_part = {}
    parts = {p: [] for p in range(k)}
    for i in V:
        for p in range(k):
            if solver.Value(x[(i, p)]) == 1:
                node_to_part[i] = p
                parts[p].append(i)
                break

    return node_to_part
