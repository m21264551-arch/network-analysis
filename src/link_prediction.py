"""
Predict future connections in a network.

The module compares classical link-prediction heuristics with a supervised
Gradient Boosting model trained on node-pair features.
"""

from typing import Iterable, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# Train and test split

def train_test_split_edges(
    G: nx.Graph,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[nx.Graph, list, list]:
    """
    Hold out existing edges for testing and sample true non-edges as negatives.

    Removed positive edges preserve the original number of connected components.
    Negative examples are sampled from non-edges in the original graph, so a
    held-out positive edge can never be mislabeled as a negative.
    """
    if G.is_directed():
        raise ValueError("train_test_split_edges expects an undirected graph")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if G.number_of_edges() == 0:
        raise ValueError("graph must contain at least one edge")

    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))
    if not non_edges:
        raise ValueError("graph has no non-edges available for negative sampling")

    rng.shuffle(edges)
    rng.shuffle(non_edges)

    target_test_count = min(max(1, int(round(len(edges) * test_fraction))), len(non_edges))
    original_components = nx.number_connected_components(G)
    G_train = G.copy()
    test_edges = []

    for u, v in edges:
        if len(test_edges) >= target_test_count:
            break
        G_train.remove_edge(u, v)
        if nx.number_connected_components(G_train) == original_components:
            test_edges.append((u, v))
        else:
            G_train.add_edge(u, v)

    if not test_edges:
        raise ValueError("could not hold out any edges without changing connectivity")

    test_non_edges = non_edges[: len(test_edges)]
    return G_train, test_edges, test_non_edges


# Heuristic predictors

def evaluate_heuristics(
    G_train: nx.Graph,
    test_edges: list,
    test_non_edges: list,
) -> pd.DataFrame:
    """
    Evaluate classical link-prediction heuristics on the held-out test set.

    The positive and negative examples must both be absent from ``G_train``.
    """
    _validate_test_edges(test_edges, test_non_edges)

    all_pairs = test_edges + test_non_edges
    y_true = np.array([1] * len(test_edges) + [0] * len(test_non_edges))

    score_sets = {
        "Common Neighbors": _common_neighbor_scores(G_train, all_pairs),
        "Jaccard Coefficient": [
            p for _, _, p in nx.jaccard_coefficient(G_train, all_pairs)
        ],
        "Adamic-Adar": [p for _, _, p in nx.adamic_adar_index(G_train, all_pairs)],
        "Preferential Attachment": [
            p for _, _, p in nx.preferential_attachment(G_train, all_pairs)
        ],
    }

    rows = []
    for name, scores in score_sets.items():
        rows.append(
            {
                "method": name,
                "AUC-ROC": round(roc_auc_score(y_true, scores), 4),
                "Avg Precision": round(average_precision_score(y_true, scores), 4),
            }
        )

    return pd.DataFrame(rows)


# Supervised model

FEATURE_NAMES = [
    "common_neighbors",
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
    """Build a feature matrix and labels for supervised link prediction."""
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
    seed: int = 42,
    max_train_edges: int = 1000,
) -> dict:
    """
    Train a Gradient Boosting classifier and evaluate it on held-out edges.

    Training uses observed edges from ``G_train`` plus sampled non-edges. Test
    metrics are computed only on ``test_edges`` and ``test_non_edges``.
    """
    _validate_test_edges(test_edges, test_non_edges)
    if max_train_edges <= 0:
        raise ValueError("max_train_edges must be positive")

    rng = np.random.default_rng(seed)
    train_pos = list(G_train.edges())
    if len(train_pos) > max_train_edges:
        rng.shuffle(train_pos)
        train_pos = train_pos[:max_train_edges]

    excluded = list(test_edges) + list(test_non_edges)
    train_neg = _sample_negative_edges(G_train, len(train_pos), rng, excluded)

    if len(train_neg) < 2 or len(train_pos) < 2:
        raise ValueError("not enough training examples for supervised link prediction")
    if len(train_neg) < len(train_pos):
        rng.shuffle(train_pos)
        train_pos = train_pos[: len(train_neg)]

    X_train, y_train = build_ml_dataset(G_train, train_pos, train_neg)
    X_test, y_test = build_ml_dataset(G_train, test_edges, test_non_edges)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )

    cv_auc = _cross_validated_auc(clf, X_train, y_train, seed)
    clf.fit(X_train, y_train)
    test_probs = clf.predict_proba(X_test)[:, 1]

    importances = pd.Series(
        clf.feature_importances_,
        index=FEATURE_NAMES,
    ).sort_values(ascending=False)

    return {
        "test_auc": round(roc_auc_score(y_test, test_probs), 4),
        "test_average_precision": round(average_precision_score(y_test, test_probs), 4),
        "cv_mean_auc": round(float(np.mean(cv_auc)), 4) if cv_auc else None,
        "cv_std_auc": round(float(np.std(cv_auc)), 4) if cv_auc else None,
        "feature_importances": importances,
        "model": clf,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }


def _pair_features(G: nx.Graph, u: int, v: int) -> list:
    """Compute features for a candidate node pair with the candidate edge absent."""
    had_edge = G.has_edge(u, v)
    edge_data = dict(G.get_edge_data(u, v, default={})) if had_edge else {}
    if had_edge:
        G.remove_edge(u, v)

    try:
        common_neighbors = len(list(nx.common_neighbors(G, u, v)))
        u_degree = G.degree(u)
        v_degree = G.degree(v)
        pref_attachment = u_degree * v_degree
        u_clustering = nx.clustering(G, u)
        v_clustering = nx.clustering(G, v)

        try:
            shortest_path = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            shortest_path = G.number_of_nodes()
    finally:
        if had_edge:
            G.add_edge(u, v, **edge_data)

    return [
        common_neighbors,
        pref_attachment,
        u_degree,
        v_degree,
        u_clustering,
        v_clustering,
        shortest_path,
    ]


def _common_neighbor_scores(G: nx.Graph, pairs: Iterable[tuple]) -> list:
    """Score pairs with the count of shared neighbours."""
    return [len(list(nx.common_neighbors(G, u, v))) for u, v in pairs]


def _sample_negative_edges(
    G: nx.Graph,
    count: int,
    rng: np.random.Generator,
    excluded_edges: Iterable[tuple],
) -> list:
    """Sample non-edges while respecting an exclusion list."""
    excluded = {_edge_key(u, v) for u, v in excluded_edges}
    candidates = [
        (u, v)
        for u, v in nx.non_edges(G)
        if _edge_key(u, v) not in excluded
    ]
    rng.shuffle(candidates)
    return candidates[:count]


def _cross_validated_auc(
    clf: GradientBoostingClassifier,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> list:
    """Run stratified cross-validation when each class has enough samples."""
    class_counts = np.bincount(y)
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        return []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auc_scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y[val_idx], probs))
    return auc_scores


def _validate_test_edges(test_edges: list, test_non_edges: list) -> None:
    """Validate the held-out test set shape."""
    if not test_edges or not test_non_edges:
        raise ValueError("test_edges and test_non_edges must be non-empty")
    if len(test_edges) != len(test_non_edges):
        raise ValueError("test_edges and test_non_edges must be balanced")


def _edge_key(u, v) -> frozenset:
    """Return an order-independent key for an undirected edge."""
    return frozenset((u, v))
