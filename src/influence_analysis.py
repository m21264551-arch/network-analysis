"""
influence_analysis.py — Centrality metrics and influence propagation simulation.

Identifies the most influential nodes using multiple centrality measures,
then simulates how information or influence spreads through the network
using the Independent Cascade (IC) model.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional


# ──────────────────────────────────────────────
# Centrality Analysis
# ──────────────────────────────────────────────

def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Compute four centrality measures for every node.
    
    Metrics
    -------
    - Degree Centrality: fraction of nodes each node is connected to
    - Betweenness Centrality: frequency a node lies on shortest paths
    - Eigenvector Centrality: influence based on connection quality
    - PageRank: random-walk importance (Google's original algorithm)
    
    Returns a DataFrame sorted by PageRank (descending).
    """
    centralities = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
        "pagerank": nx.pagerank(G),
    }

    df = pd.DataFrame(centralities)
    df.index.name = "node"
    df = df.sort_values("pagerank", ascending=False)
    return df


def get_top_influencers(
    centrality_df: pd.DataFrame, metric: str = "pagerank", top_n: int = 10
) -> pd.DataFrame:
    """Return the top-N nodes by a given centrality metric."""
    return centrality_df.nlargest(top_n, metric)


def centrality_correlation(centrality_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation between centrality measures."""
    return centrality_df.corr(method="spearman").round(4)


# ──────────────────────────────────────────────
# Influence Propagation — Independent Cascade
# ──────────────────────────────────────────────

def independent_cascade(
    G: nx.Graph,
    seed_nodes: List[int],
    propagation_prob: float = 0.1,
    max_steps: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Simulate the Independent Cascade (IC) model.
    
    At each step, every newly activated node has one chance to activate
    each inactive neighbour with probability `propagation_prob`.
    
    Parameters
    ----------
    G : nx.Graph
    seed_nodes : list – Initially activated ("infected") nodes
    propagation_prob : float – Per-edge activation probability
    max_steps : int – Halt after this many steps even if spreading
    
    Returns
    -------
    dict with keys:
        'activated' : set of all activated nodes
        'history'   : list of sets (nodes activated at each step)
        'reach'     : fraction of total network activated
    """
    if rng is None:
        rng = np.random.default_rng(42)

    activated = set(seed_nodes)
    history = [set(seed_nodes)]
    newly_activated = set(seed_nodes)

    for step in range(max_steps):
        next_activated = set()
        for node in newly_activated:
            for neighbour in G.neighbors(node):
                if neighbour not in activated:
                    if rng.random() < propagation_prob:
                        next_activated.add(neighbour)

        if not next_activated:
            break

        activated |= next_activated
        history.append(next_activated)
        newly_activated = next_activated

    return {
        "activated": activated,
        "history": history,
        "reach": len(activated) / G.number_of_nodes(),
        "steps": len(history),
    }


def compare_seed_strategies(
    G: nx.Graph,
    centrality_df: pd.DataFrame,
    seed_size: int = 5,
    propagation_prob: float = 0.1,
    num_simulations: int = 100,
) -> pd.DataFrame:
    """
    Compare different seed-selection strategies for influence maximisation.
    
    Strategies
    ----------
    - Top PageRank nodes
    - Top Betweenness nodes
    - Top Degree nodes
    - Random nodes (baseline)
    
    Runs multiple simulations and reports mean ± std reach.
    """
    rng = np.random.default_rng(42)
    all_nodes = list(G.nodes())

    strategies = {
        "PageRank": centrality_df.nlargest(seed_size, "pagerank").index.tolist(),
        "Betweenness": centrality_df.nlargest(seed_size, "betweenness").index.tolist(),
        "Degree": centrality_df.nlargest(seed_size, "degree").index.tolist(),
    }

    results = []
    for name, seeds in strategies.items():
        reaches = []
        for _ in range(num_simulations):
            sim = independent_cascade(G, seeds, propagation_prob, rng=rng)
            reaches.append(sim["reach"])
        results.append({
            "strategy": name,
            "mean_reach": round(np.mean(reaches), 4),
            "std_reach": round(np.std(reaches), 4),
            "seeds": seeds,
        })

    # Random baseline
    random_reaches = []
    for _ in range(num_simulations):
        random_seeds = rng.choice(all_nodes, size=seed_size, replace=False).tolist()
        sim = independent_cascade(G, random_seeds, propagation_prob, rng=rng)
        random_reaches.append(sim["reach"])
    results.append({
        "strategy": "Random",
        "mean_reach": round(np.mean(random_reaches), 4),
        "std_reach": round(np.std(random_reaches), 4),
        "seeds": "random",
    })

    return pd.DataFrame(results)
