"""
network_builder.py — Constructs and loads social network graphs.

Supports loading real-world datasets (Karate Club, Les Misérables)
and generating synthetic scale-free networks with planted community structure.
"""

import networkx as nx
import numpy as np
from typing import Optional


def load_karate_club() -> nx.Graph:
    """Load Zachary's Karate Club — classic community detection benchmark."""
    G = nx.karate_club_graph()
    # Store ground-truth communities
    for node in G.nodes():
        G.nodes[node]["ground_truth"] = G.nodes[node]["club"]
    return G


def load_les_miserables() -> nx.Graph:
    """Load Les Misérables character co-appearance network."""
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
    
    This synthetic network has realistic degree distributions and 
    planted community structure — ideal for evaluating detection algorithms.
    
    Parameters
    ----------
    n : int – Number of nodes
    tau1 : float – Power-law exponent for degree distribution
    tau2 : float – Power-law exponent for community size distribution
    mu : float – Mixing parameter (0 = perfect communities, 1 = random)
    seed : int – Random seed for reproducibility
    
    Returns
    -------
    nx.Graph with 'community' node attribute
    """
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
    # Convert frozenset communities to integer labels
    communities = {node: list(data["community"])[0] for node, data in G.nodes(data=True)}
    comm_map = {}
    for node, comm in communities.items():
        if comm not in comm_map:
            comm_map[comm] = len(comm_map)
        G.nodes[node]["ground_truth"] = comm_map[comm]

    # Remove the frozenset attribute (not serialisable)
    for node in G.nodes():
        if "community" in G.nodes[node]:
            del G.nodes[node]["community"]

    return G


def generate_stochastic_block_model(
    sizes: Optional[list] = None,
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

    k = len(sizes)
    probs = [[p_intra if i == j else p_inter for j in range(k)] for i in range(k)]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    # Label ground truth
    node_idx = 0
    for comm_id, size in enumerate(sizes):
        for _ in range(size):
            G.nodes[node_idx]["ground_truth"] = comm_id
            node_idx += 1

    return G


def get_network_summary(G: nx.Graph) -> dict:
    """Return a summary dict of key network statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "is_connected": nx.is_connected(G),
        "num_components": nx.number_connected_components(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
    }
