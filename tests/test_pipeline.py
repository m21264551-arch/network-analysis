import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from main import _balanced_community_sizes
from src.community_detection import louvain
from src.influence_analysis import compute_centralities, independent_cascade
from src.link_prediction import (
    evaluate_heuristics,
    train_link_predictor,
    train_test_split_edges,
)
from src.network_builder import generate_stochastic_block_model, get_network_summary
from src.visualisation import plot_degree_distribution


def test_balanced_community_sizes_preserves_requested_node_count():
    sizes = _balanced_community_sizes(302, 5)

    assert sizes == [61, 61, 60, 60, 60]
    assert sum(sizes) == 302


def test_network_summary_handles_empty_graph():
    summary = get_network_summary(nx.Graph())

    assert summary["nodes"] == 0
    assert summary["avg_degree"] == 0
    assert summary["num_components"] == 0


def test_louvain_uses_networkx_implementation():
    graph = nx.karate_club_graph()
    partition = louvain(graph)

    assert set(partition) == set(graph.nodes())
    assert len(set(partition.values())) >= 2


def test_link_split_uses_true_non_edges_for_negatives():
    graph = nx.karate_club_graph()
    original_edges = {_edge_key(u, v) for u, v in graph.edges()}

    train_graph, test_pos, test_neg = train_test_split_edges(graph, test_fraction=0.15)

    assert len(test_pos) == len(test_neg)
    assert all(_edge_key(u, v) in original_edges for u, v in test_pos)
    assert all(_edge_key(u, v) not in original_edges for u, v in test_neg)
    assert all(not train_graph.has_edge(u, v) for u, v in test_pos)


def test_link_prediction_reports_holdout_metrics():
    graph = generate_stochastic_block_model(
        sizes=[12, 12, 12],
        p_intra=0.35,
        p_inter=0.03,
        seed=7,
    )
    train_graph, test_pos, test_neg = train_test_split_edges(graph, test_fraction=0.2)

    heuristic_results = evaluate_heuristics(train_graph, test_pos, test_neg)
    ml_results = train_link_predictor(train_graph, test_pos, test_neg)

    assert set(heuristic_results["method"]) == {
        "Common Neighbors",
        "Jaccard Coefficient",
        "Adamic-Adar",
        "Preferential Attachment",
    }
    assert 0 <= ml_results["test_auc"] <= 1
    assert 0 <= ml_results["test_average_precision"] <= 1
    assert ml_results["train_samples"] > ml_results["test_samples"]


def test_independent_cascade_validates_inputs():
    graph = nx.path_graph(3)

    with pytest.raises(ValueError, match="propagation_prob"):
        independent_cascade(graph, [0], propagation_prob=1.5)

    with pytest.raises(ValueError, match="Seed nodes"):
        independent_cascade(graph, [99])


def test_degree_distribution_handles_isolates():
    graph = nx.empty_graph(4)

    fig = plot_degree_distribution(graph)

    assert fig.axes
    plt.close(fig)


def test_compute_centralities_handles_empty_graph():
    centralities = compute_centralities(nx.Graph())

    assert centralities.empty
    assert list(centralities.columns) == [
        "degree",
        "betweenness",
        "eigenvector",
        "pagerank",
    ]


def _edge_key(u, v):
    return frozenset((u, v))
