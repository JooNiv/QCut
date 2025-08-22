from sklearn.cluster import KMeans
import rustworkx as rx
import numpy as np


def weight_fn(edge_data):
    """
    Custom weight function for edges.
    """
    if isinstance(edge_data, tuple):
        return edge_data[3]  # Use the weight from the tuple
    return 4.0  # Default weight


def k_way_kmeans_partition(
    graph: rx.PyGraph,
    k: int,
    random_state: int = np.random.randint(0, 10000),
) -> np.ndarray:
    """
    Partition an undirected graph into k parts using KMeans clustering.

    Args:
        graph: A rustworkx.PyGraph instance (undirected).
        k: Number of partitions.
        random_state: Seed for k-means.

    Returns:
        labels: A numpy array of length n_nodes where each entry is in [0, k-1] indicating partition assignment.
    """
    # Build adjacency matrix
    adj = rx.adjacency_matrix(graph, weight_fn=weight_fn, default_weight=4.0)

    # Use KMeans on the adjacency matrix
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(adj)

    node_to_label = {node: int(label) for node, label in zip(graph.nodes(), labels)}

    print(labels)

    return node_to_label
