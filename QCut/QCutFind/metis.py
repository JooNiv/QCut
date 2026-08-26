"""
The Metis graph partitioning utility for QCut.
"""

from collections import defaultdict

import pymetis
import rustworkx as rx

from QCut.QCutFind.graph_circuit_utils import weight_fn


def build_csr(graph: rx.PyGraph, weight_fn):
    """
    Build METIS-style CSR (xadj, adjncy, eweights) in a single O(n + m) pass.
    """
    n = graph.num_nodes()
    # initialize an adjacency list
    nbr_map = defaultdict(list)  # node → list of (neighbor, weight)

    # one pass over edges
    for u, v, data in graph.weighted_edge_list():
        w = weight_fn(data)
        nbr_map[u].append((v, w))
        nbr_map[v].append((u, w))

    xadj = [0] * (n + 1)
    adjncy = []
    eweights = []

    offset = 0
    for i in range(n):
        neighs = nbr_map.get(i, [])
        xadj[i] = offset
        for v, w in neighs:
            adjncy.append(v)
            eweights.append(w)
            offset += 1
    xadj[n] = offset

    return xadj, adjncy, eweights


#: Weight a qubit's nodes sum to. Only the ratio to the other qubits matters, so this
#: just has to be large enough that the per-node integer share does not round away.
QUBIT_SCALE: int = 1000

#: Imbalance METIS is allowed when it is balancing qubits against a budget, in units of
#: 0.1%. Tight, because the budget is a hard constraint and anything left over has to be
#: repaired afterwards by moving whole qubits, which costs cuts.
BUDGET_UFACTOR: int = 1

#: Imbalance allowed when there is no budget. Wide on purpose: an unbalanced split is
#: often much cheaper, and with nothing to satisfy there is no reason to refuse it.
FREE_UFACTOR: int = 500


def qubit_node_weights(graph: rx.PyGraph, nodes_on_qubit: dict) -> list[int]:
    """Vertex weights that make a partition's weight count its qubits.

    The graph's nodes are wire segments, so several belong to one qubit and balancing
    node counts says nothing about balancing qubits. Giving each of a qubit's nodes an
    equal share of one qubit's worth makes a partition's total weight the number of
    qubits it holds, counting a qubit that straddles the cut in both -- which is what a
    wire cut costs anyway.
    """
    weights = [1] * graph.num_nodes()
    for nodes in nodes_on_qubit.values():
        if not nodes:
            continue
        share = max(1, round(QUBIT_SCALE / len(nodes)))
        for node in nodes:
            weights[node] = share
    return weights


def k_way_metis_partition(
    graph: rx.PyGraph,
    k: int,
    seed: int = 0,
    ncuts: int = 5,
    node_weights: list[int] | None = None,
    targets: list[float] | None = None,
    ufactor: int = FREE_UFACTOR,
):
    """Partition ``graph`` into ``k`` parts, minimising the weighted edge cut.

    ``seed`` used to be drawn at random, which made the whole cut finder
    non-deterministic: two runs on one circuit could return partitions whose sampling
    overheads differed by three orders of magnitude, with no way to reproduce the good
    one. It is now the caller's to choose.

    ``ncuts`` is how many partitionings METIS tries internally before returning the best
    it saw. Its notion of best is the weighted edge cut, which is the true cost only
    while no cuts are being decomposed jointly, so a caller that can score a whole plan
    is better off asking for one partitioning per seed and comparing the plans itself.

    ``node_weights`` and ``targets`` are how a qubit budget is expressed: weights from
    :func:`qubit_node_weights` make a partition's weight its qubit count, and targets
    say what share of them each partition may hold. Without them METIS balances node
    counts, which is unrelated to the budget, and the result has to be repaired
    afterwards by moving whole qubits across -- which is far more expensive than asking
    for the right thing in the first place.
    """
    n = graph.num_nodes()

    xadj, adjncy, eweights = build_csr(graph, weight_fn)

    options = pymetis.Options(
        ncuts=ncuts,
        nseps=5,
        numbering=-1,
        niter=20,
        minconn=0,
        no2hop=0,
        seed=seed,
        contig=1,
        compress=0,
        ccorder=0,
        pfactor=0,
        ufactor=ufactor,
    )

    obj_val, parts = pymetis.part_graph(
        nparts=k,
        # The same CSR arrays build_csr already produces, just handed over as the
        # object pymetis wants: passing xadj/adjncy directly is deprecated.
        adjacency=pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy),
        vweights=node_weights,
        eweights=eweights,  # adjwgt (edge weights)
        tpwgts=targets,  # target share per partition
        recursive=True,
        contiguous=None,
        options=options,
    )

    return {i: parts[i] for i in range(n)}
