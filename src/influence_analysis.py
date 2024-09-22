"""
Compute centrality metrics and simulate influence propagation.

The module identifies influential nodes with standard centrality measures and
uses the Independent Cascade model to estimate network reach.
"""

from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd


# Centrality analysis

CENTRALITY_COLUMNS = ["degree", "betweenness", "eigenvector", "pagerank"]


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """
    Compute centrality measures for every node.

    Returns a DataFrame sorted by PageRank in descending order.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=CENTRALITY_COLUMNS).rename_axis("node")

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G)

    centralities = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "eigenvector": eigenvector,
        "pagerank": nx.pagerank(G),
    }

    df = pd.DataFrame(centralities)
    df.index.name = "node"
    return df.sort_values("pagerank", ascending=False)


def get_top_influencers(
    centrality_df: pd.DataFrame,
    metric: str = "pagerank",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top nodes for a centrality metric."""
    if metric not in centrality_df.columns:
        raise ValueError(f"Unknown centrality metric: {metric}")
    return centrality_df.nlargest(top_n, metric)


def centrality_correlation(centrality_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank correlation between centrality measures."""
    return centrality_df.corr(method="spearman").round(4)


# Influence propagation

def independent_cascade(
    G: nx.Graph,
    seed_nodes: List[int],
    propagation_prob: float = 0.1,
    max_steps: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Simulate the Independent Cascade model.

    Each newly activated node gets one chance to activate each inactive
    neighbour with probability ``propagation_prob``.
    """
    if not 0 <= propagation_prob <= 1:
        raise ValueError("propagation_prob must be between 0 and 1")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    seed_nodes = list(dict.fromkeys(seed_nodes))
    missing = set(seed_nodes) - set(G.nodes())
    if missing:
        raise ValueError(f"Seed nodes are not in the graph: {sorted(missing)}")

    if rng is None:
        rng = np.random.default_rng(42)

    activated = set(seed_nodes)
    history = [set(seed_nodes)] if seed_nodes else []
    newly_activated = set(seed_nodes)

    for _ in range(max_steps):
        next_activated = set()
        for node in newly_activated:
            for neighbour in G.neighbors(node):
                if neighbour not in activated and rng.random() < propagation_prob:
                    next_activated.add(neighbour)

        if not next_activated:
            break

        activated |= next_activated
        history.append(next_activated)
        newly_activated = next_activated

    node_count = G.number_of_nodes()
    reach = len(activated) / node_count if node_count else 0
    return {
        "activated": activated,
        "history": history,
        "reach": reach,
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
    Compare seed-selection strategies for influence maximisation.

    The result reports the mean and standard deviation of network reach across
    repeated simulations.
    """
    if seed_size < 0:
        raise ValueError("seed_size must be non-negative")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive")

    all_nodes = list(G.nodes())
    if not all_nodes:
        return pd.DataFrame(
            columns=["strategy", "mean_reach", "std_reach", "seeds"]
        )

    seed_size = min(seed_size, len(all_nodes))
    rng = np.random.default_rng(42)

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
        results.append(
            {
                "strategy": name,
                "mean_reach": round(np.mean(reaches), 4),
                "std_reach": round(np.std(reaches), 4),
                "seeds": seeds,
            }
        )

    random_reaches = []
    for _ in range(num_simulations):
        random_seeds = rng.choice(all_nodes, size=seed_size, replace=False).tolist()
        sim = independent_cascade(G, random_seeds, propagation_prob, rng=rng)
        random_reaches.append(sim["reach"])
    results.append(
        {
            "strategy": "Random",
            "mean_reach": round(np.mean(random_reaches), 4),
            "std_reach": round(np.std(random_reaches), 4),
            "seeds": "random",
        }
    )

    return pd.DataFrame(results)
