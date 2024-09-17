"""
community_detection.py — Multiple community detection algorithms with evaluation.

Implements Louvain, Label Propagation, Girvan-Newman, and spectral clustering,
then evaluates results against ground truth using NMI and modularity.
"""

import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import community as community_louvain  # python-louvain
except ImportError:
    community_louvain = None

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ──────────────────────────────────────────────
# Detection Algorithms
# ──────────────────────────────────────────────

def louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> Dict[int, int]:
    """
    Louvain method for community detection.
    
    Greedy modularity optimisation — fast, scalable, widely used in industry.
    Returns dict mapping node → community_id.
    """
    if community_louvain is None:
        raise ImportError("Install python-louvain: pip install python-louvain")
    return community_louvain.best_partition(G, resolution=resolution, random_state=seed)


def label_propagation(G: nx.Graph) -> Dict[int, int]:
    """
    Label Propagation Algorithm (LPA).
    
    Near-linear time, no parameters — each node adopts the majority
    label of its neighbours until convergence.
    """
    communities = nx.algorithms.community.label_propagation_communities(G)
    partition = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            partition[node] = comm_id
    return partition


def girvan_newman(G: nx.Graph, k: int = 5) -> Dict[int, int]:
    """
    Girvan-Newman algorithm — iteratively removes highest-betweenness edges.
    
    Parameters
    ----------
    k : int – Target number of communities
    """
    comp = nx.algorithms.community.girvan_newman(G)
    for communities in comp:
        if len(communities) >= k:
            break

    partition = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            partition[node] = comm_id
    return partition


def spectral_clustering(G: nx.Graph, k: int = 5) -> Dict[int, int]:
    """
    Spectral clustering on the graph Laplacian.
    
    Uses eigenvectors of the normalised Laplacian + k-means.
    """
    from sklearn.cluster import KMeans

    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Take the first k non-trivial eigenvectors
    features = eigenvectors[:, 1 : k + 1]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    nodes = list(G.nodes())
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_partition(
    G: nx.Graph, partition: Dict[int, int], ground_truth_attr: str = "ground_truth"
) -> dict:
    """
    Evaluate a community partition against ground truth.
    
    Metrics
    -------
    - Modularity (no ground truth needed)
    - Normalised Mutual Information (NMI)
    - Adjusted Rand Index (ARI)
    """
    # Modularity
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    modularity = nx.algorithms.community.modularity(G, communities.values())

    # Ground-truth comparison
    gt = nx.get_node_attributes(G, ground_truth_attr)
    if gt:
        nodes = sorted(set(partition.keys()) & set(gt.keys()))
        pred = [partition[n] for n in nodes]
        true = [gt[n] for n in nodes]
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
    Run all community detection algorithms and return partitions + metrics.
    
    Returns
    -------
    dict mapping algorithm_name → (partition_dict, evaluation_metrics)
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
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")

    return results
