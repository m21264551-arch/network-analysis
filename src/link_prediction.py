"""
link_prediction.py — Predict future connections in the network.

Compares heuristic-based predictors (Common Neighbours, Jaccard, Adamic-Adar,
Preferential Attachment) with a supervised ML approach using node-pair features.
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────
# Train / Test Split on Edges
# ──────────────────────────────────────────────

def train_test_split_edges(
    G: nx.Graph, test_fraction: float = 0.2, seed: int = 42
) -> Tuple[nx.Graph, list, list]:
    """
    Hold out a fraction of edges for testing.
    
    Returns
    -------
    G_train : Graph with test edges removed (stays connected)
    test_edges : list of held-out positive edges
    test_non_edges : list of sampled negative edges
    """
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    rng.shuffle(edges)

    n_test = int(len(edges) * test_fraction)
    G_train = G.copy()
    test_edges = []

    # Remove edges but keep graph connected
    for u, v in edges:
        if len(test_edges) >= n_test:
            break
        G_train.remove_edge(u, v)
        if nx.is_connected(G_train):
            test_edges.append((u, v))
        else:
            G_train.add_edge(u, v)

    # Sample negative edges
    non_edges = list(nx.non_edges(G_train))
    rng.shuffle(non_edges)
    test_non_edges = non_edges[: len(test_edges)]

    return G_train, test_edges, test_non_edges


# ──────────────────────────────────────────────
# Heuristic Predictors
# ──────────────────────────────────────────────

def evaluate_heuristics(
    G_train: nx.Graph,
    test_edges: list,
    test_non_edges: list,
) -> pd.DataFrame:
    """
    Evaluate classical link prediction heuristics.
    
    Heuristics
    ----------
    - Common Neighbours
    - Jaccard Coefficient
    - Adamic-Adar Index
    - Preferential Attachment
    """
    all_pairs = test_edges + test_non_edges
    y_true = [1] * len(test_edges) + [0] * len(test_non_edges)

    heuristics = {
        "Common Neighbours": nx.common_neighbor_centrality(G_train, all_pairs),
        "Jaccard Coefficient": nx.jaccard_coefficient(G_train, all_pairs),
        "Adamic-Adar": nx.adamic_adar_index(G_train, all_pairs),
        "Preferential Attachment": nx.preferential_attachment(G_train, all_pairs),
    }

    results = []
    for name, preds_gen in heuristics.items():
        scores = [p for _, _, p in preds_gen]
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
        results.append({"method": name, "AUC-ROC": round(auc, 4), "Avg Precision": round(ap, 4)})

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# Supervised ML Approach
# ──────────────────────────────────────────────

def _pair_features(G: nx.Graph, u: int, v: int) -> list:
    """Compute feature vector for a node pair (u, v)."""
    cn = len(list(nx.common_neighbors(G, u, v)))
    u_deg, v_deg = G.degree(u), G.degree(v)
    pa = u_deg * v_deg
    u_cc = nx.clustering(G, u)
    v_cc = nx.clustering(G, v)

    try:
        shortest_path = nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        shortest_path = -1

    return [cn, pa, u_deg, v_deg, u_cc, v_cc, shortest_path]


FEATURE_NAMES = [
    "common_neighbours",
    "pref_attachment",
    "degree_u",
    "degree_v",
    "clustering_u",
    "clustering_v",
    "shortest_path",
]


def build_ml_dataset(
    G_train: nx.Graph,
    pos_edges: list,
    neg_edges: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and labels for supervised link prediction."""
    X, y = [], []
    for u, v in pos_edges:
        X.append(_pair_features(G_train, u, v))
        y.append(1)
    for u, v in neg_edges:
        X.append(_pair_features(G_train, u, v))
        y.append(0)
    return np.array(X), np.array(y)


def train_link_predictor(
    G_train: nx.Graph,
    test_edges: list,
    test_non_edges: list,
) -> dict:
    """
    Train a Gradient Boosting classifier for link prediction.
    
    Uses cross-validation on the test set and reports AUC-ROC + feature importances.
    """
    X, y = build_ml_dataset(G_train, test_edges, test_non_edges)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )

    # Cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y[val_idx], probs))

    # Fit final model on all data
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(
        ascending=False
    )

    return {
        "mean_auc": round(np.mean(auc_scores), 4),
        "std_auc": round(np.std(auc_scores), 4),
        "feature_importances": importances,
        "model": clf,
    }
