import numpy as np
import rustworkx as rx
from sklearn.cluster import KMeans, SpectralClustering


def weight_fn(edge_data):
    """
    Custom weight function for edges.
    """
    if isinstance(edge_data, tuple):
        return edge_data[3]  # Use the weight from the tuple
    return 4.0  # Default weight


def k_way_spectral_partition(
    graph: rx.PyGraph,
    k: int,
    random_state: int = np.random.randint(0, 10000),
) -> np.ndarray:
    """
    Partition an undirected graph into k parts using spectral clustering.

    Args:
        graph: A rustworkx.PyGraph instance (undirected).
        k: Number of partitions.
        random_state: Seed for k-means.

    Returns:
        labels: A numpy array of length n_nodes where each entry is in [0, k-1] 
        indicating partition assignment.
    """
    # Ensure graph is undirected

    n = graph.num_nodes()
    # Build adjacency matrix
    # rustworkx adjacency matrix: nodes numbered 0..n-1
    adj = rx.adjacency_matrix(graph, weight_fn=weight_fn)

    # Degree matrix
    deg = np.diag(adj.sum(axis=1))

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    # with np.errstate(divide="ignore"):
    # d_inv_sqrt = np.diag(1.0 / np.sqrt(np.where(deg.diagonal() > 0, 
    # deg.diagonal(), 1)))
    # L = np.eye(n) - d_inv_sqrt @ adj @ d_inv_sqrt
    L = deg - adj  # Unnormalized Laplacian
    # Compute first k eigenvectors of L
    # Use eigh since L is symmetric
    eigvals, eigvecs = np.linalg.eigh(L)
    # Take k smallest non-zero eigenvalues (skip the first zero eigenvalue)
    X = eigvecs[:, 1:k] if k > 1 else eigvecs[:, [1]]

    # Normalize rows for k-means
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X_norm)

    node_to_label = {node: int(label) for node, label in zip(graph.nodes(), labels)}

    print(labels)

    return node_to_label


def k_way_spectral_partition_sk(
    graph: rx.PyGraph,
    k: int,
    random_state: int = np.random.randint(0, 10000),
) -> dict[int, int]:
    """
    Partition an undirected graph into k parts using sklearn's SpectralClustering.

    Args:
        graph: A rustworkx.PyGraph instance (undirected).
        k: Number of partitions.
        random_state: Seed for the spectral embedding + k-means discretiation.

    Returns:
        node_to_label: A dict mapping each node index to its partition label in [0..k-1].
    """
    # Build the (symmetric) adjacency matrix
    # rustworkx numbers nodes 0..n-1
    adj = rx.adjacency_matrix(graph, weight_fn=weight_fn)

    # Use SpectralClustering with a precomputed affinity matrix
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=random_state,
    )
    labels = sc.fit_predict(adj)

    # Return as a dict mapping node → cluster
    return {node: int(label) for node, label in zip(graph.nodes(), labels)}


def qubits_from_graph(graph: rx.PyGraph) -> list[int]:
    """
    Extracts the qubit indices from the graph nodes.

    Args:
        graph: A rustworkx.PyGraph instance.

    Returns:
        A list of qubit indices.
    """
    return len({edge for edge in graph.edges() if isinstance(edge, int)})


def spectral_recursive(graph: rx.PyGraph, qubits_per_subcircuit: list[int] = []):
    return
