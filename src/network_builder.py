"""
Build and load social network graphs.

Supports Zachary's Karate Club, the Les Miserables character network, and
synthetic networks with planted community structure.
"""

import networkx as nx
from typing import Optional, Sequence


def load_karate_club() -> nx.Graph:
    """Load Zachary's Karate Club, a classic community detection benchmark."""
    G = nx.karate_club_graph()
    for node in G.nodes():
        G.nodes[node]["ground_truth"] = G.nodes[node]["club"]
    return G


def load_les_miserables() -> nx.Graph:
    """Load the Les Miserables character co-appearance network."""
    return nx.les_miserables_graph()


def generate_lfr_benchmark(
    n: int = 500,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.1,
    min_degree: int = 5,
    max_degree: int = 50,
    min_community: int = 20,
    max_community: int = 80,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a Lancichinetti-Fortunato-Radicchi (LFR) benchmark graph.

    This synthetic network has realistic degree distributions and planted
    community structure, which makes it useful for evaluating detection
    algorithms.

    Parameters
    ----------
    n : int
        Number of nodes.
    tau1 : float
        Power-law exponent for degree distribution.
    tau2 : float
        Power-law exponent for community size distribution.
    mu : float
        Mixing parameter where 0 is strongly clustered and 1 is random.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Graph with a ``ground_truth`` node attribute.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= mu <= 1:
        raise ValueError("mu must be between 0 and 1")

    G = nx.LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed,
    )

    # Convert community membership sets to stable integer labels.
    comm_map = {}
    for node, data in G.nodes(data=True):
        community = tuple(sorted(data["community"]))
        if community not in comm_map:
            comm_map[community] = len(comm_map)
        G.nodes[node]["ground_truth"] = comm_map[community]
        if "community" in G.nodes[node]:
            del G.nodes[node]["community"]

    return G


def generate_stochastic_block_model(
    sizes: Optional[Sequence[int]] = None,
    p_intra: float = 0.25,
    p_inter: float = 0.01,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a Stochastic Block Model (SBM) graph.

    A simpler planted-partition model useful for controlled experiments.
    """
    if sizes is None:
        sizes = [60, 80, 50, 70, 40]
    if not sizes:
        raise ValueError("sizes must contain at least one community")
    if any(size <= 0 for size in sizes):
        raise ValueError("all community sizes must be positive")
    if not 0 <= p_intra <= 1 or not 0 <= p_inter <= 1:
        raise ValueError("edge probabilities must be between 0 and 1")

    sizes = list(sizes)

    k = len(sizes)
    probs = [[p_intra if i == j else p_inter for j in range(k)] for i in range(k)]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    node_idx = 0
    for comm_id, size in enumerate(sizes):
        for _ in range(size):
            G.nodes[node_idx]["ground_truth"] = comm_id
            node_idx += 1

    return G


def get_network_summary(G: nx.Graph) -> dict:
    """Return a summary dict of key network statistics."""
    node_count = G.number_of_nodes()
    if node_count == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "avg_clustering": 0,
            "is_connected": False,
            "num_components": 0,
            "avg_degree": 0,
        }

    num_components = nx.number_connected_components(G)
    return {
        "nodes": node_count,
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "is_connected": num_components == 1,
        "num_components": num_components,
        "avg_degree": sum(dict(G.degree()).values()) / node_count,
    }
