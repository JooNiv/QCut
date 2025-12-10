from collections import defaultdict

import numpy as np
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


def k_way_metis_partition(graph: rx.PyGraph, k: int):
    n = graph.num_nodes()

    xadj, adjncy, eweights = build_csr(graph, weight_fn)

    options = pymetis.Options(
        ncuts=5,
        nseps=5,
        numbering=-1,
        niter=20,
        minconn=0,
        no2hop=0,
        seed=np.random.randint(0, 10000),
        contig=1,
        compress=0,
        ccorder=0,
        pfactor=0,
        ufactor=500,
    )

    obj_val, parts = pymetis.part_graph(
        k,  # nparts
        None,  # adjacency (pythonic)
        xadj,  # xadj
        adjncy,  # adjncy
        None,  # vwgt
        eweights,  # adjwgt (edge weights)
        True,  # recursive
        None,  # contiguous
        options,  # options
    )

    return {i: parts[i] for i in range(n)}
