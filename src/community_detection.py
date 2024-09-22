"""
Run community detection algorithms and evaluate their partitions.

Includes Louvain, Label Propagation, Girvan-Newman, and spectral clustering.
Partitions are evaluated with modularity, NMI, and ARI when ground truth labels
are available.
"""

from collections import defaultdict
from typing import Dict, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Detection algorithms

def louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> Dict[int, int]:
    """
    Run Louvain community detection.

    This uses NetworkX's built-in implementation, so the project does not need
    a separate python-louvain dependency.
    """
    communities = nx.algorithms.community.louvain_communities(
        G,
        resolution=resolution,
        seed=seed,
    )
    return _communities_to_partition(communities)


def label_propagation(G: nx.Graph) -> Dict[int, int]:
    """
    Run Label Propagation.

    This near-linear algorithm has no tuning parameters. Each node adopts the
    majority label of its neighbours until the labels stabilise.
    """
    communities = nx.algorithms.community.label_propagation_communities(G)
    return _communities_to_partition(communities)


def girvan_newman(G: nx.Graph, k: int = 5) -> Dict[int, int]:
    """Run Girvan-Newman until at least ``k`` communities are found."""
    target = _normalise_k(G, k)
    if target == 0:
        return {}
    if target == 1:
        return {node: 0 for node in G.nodes()}
    if G.number_of_edges() == 0:
        return {node: idx for idx, node in enumerate(G.nodes())}

    communities = tuple(nx.connected_components(G))
    if len(communities) >= target:
        return _communities_to_partition(communities)

    for communities in nx.algorithms.community.girvan_newman(G):
        if len(communities) >= target:
            break

    return _communities_to_partition(communities)


def spectral_clustering(G: nx.Graph, k: int = 5) -> Dict[int, int]:
    """
    Run spectral clustering on the graph Laplacian.

    The graph is embedded with the smallest non-trivial eigenvectors of the
    normalised Laplacian, then clustered with k-means.
    """
    from sklearn.cluster import KMeans

    target = _normalise_k(G, k)
    if target == 0:
        return {}
    if target == 1:
        return {node: 0 for node in G.nodes()}

    laplacian = nx.normalized_laplacian_matrix(G).toarray()
    _, eigenvectors = np.linalg.eigh(laplacian)
    features = eigenvectors[:, 1 : target + 1]

    kmeans = KMeans(n_clusters=target, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    nodes = list(G.nodes())
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}


# Evaluation

def evaluate_partition(
    G: nx.Graph,
    partition: Dict[int, int],
    ground_truth_attr: str = "ground_truth",
) -> dict:
    """
    Evaluate a community partition against ground truth when labels exist.

    Metrics include modularity, Normalised Mutual Information, and Adjusted
    Rand Index.
    """
    communities = defaultdict(set)
    for node in G.nodes():
        comm = partition.get(node, f"singleton_{node}")
        communities[comm].add(node)

    modularity = (
        nx.algorithms.community.modularity(G, communities.values())
        if G.number_of_edges() > 0 and communities
        else 0
    )

    ground_truth = nx.get_node_attributes(G, ground_truth_attr)
    if ground_truth:
        nodes = sorted(set(partition.keys()) & set(ground_truth.keys()))
        pred = [partition[n] for n in nodes]
        true = [ground_truth[n] for n in nodes]
        nmi = normalized_mutual_info_score(true, pred)
        ari = adjusted_rand_score(true, pred)
    else:
        nmi, ari = None, None

    return {
        "num_communities": len(communities),
        "modularity": round(modularity, 4),
        "nmi": round(nmi, 4) if nmi is not None else None,
        "ari": round(ari, 4) if ari is not None else None,
    }


def run_all_detectors(G: nx.Graph, k: int = 5) -> Dict[str, Tuple[Dict, dict]]:
    """
    Run every detector and return partitions with evaluation metrics.

    Returns a dictionary mapping algorithm names to ``(partition, metrics)``.
    """
    results = {}
    detectors = {
        "Louvain": lambda: louvain(G),
        "Label Propagation": lambda: label_propagation(G),
        "Girvan-Newman": lambda: girvan_newman(G, k=k),
        "Spectral Clustering": lambda: spectral_clustering(G, k=k),
    }

    for name, detect_fn in detectors.items():
        try:
            partition = detect_fn()
            metrics = evaluate_partition(G, partition)
            results[name] = (partition, metrics)
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")

    return results


def _normalise_k(G: nx.Graph, k: int) -> int:
    """Clamp a requested community count to a valid range for the graph."""
    if G.number_of_nodes() == 0:
        return 0
    return min(max(1, int(k)), G.number_of_nodes())


def _communities_to_partition(communities) -> Dict[int, int]:
    """Convert an iterable of node communities into a node partition mapping."""
    partition = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            partition[node] = comm_id
    return partition
