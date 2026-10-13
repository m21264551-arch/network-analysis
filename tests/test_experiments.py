import json
from pathlib import Path
import networkx as nx
import numpy as np
import pytest
from src.experiments import build_experiment, paired_cascade
from src.link_prediction import _pair_features, train_test_split_edges, build_ml_dataset


def test_pair_features_are_symmetric_and_preserve_graph():
    g = nx.karate_club_graph()
    before = nx.node_link_data(g, edges="links")
    for u, v in [(0, 1), (0, 33), (14, 20)]:
        assert _pair_features(g, u, v) == _pair_features(g, v, u)
    assert {frozenset((u, v)): d for u, v, d in g.edges(data=True)} == {
        frozenset((e["source"], e["target"])): {
            k: v for k, v in e.items() if k not in ["source", "target"]
        }
        for e in before["links"]
    }


def test_feature_extraction_never_restores_held_out_edges():
    g = nx.karate_club_graph()
    train, pos, neg = train_test_split_edges(g, 0.15, 3)
    before = {frozenset(e) for e in train.edges}
    features, _ = build_ml_dataset(train, pos, neg)
    assert len(features) == 2 * len(pos)
    assert all(not train.has_edge(*e) for e in pos)
    assert before == {frozenset(e) for e in train.edges}
    assert nx.number_connected_components(g) == nx.number_connected_components(train)
    assert all(train.edges[u, v] == g.edges[u, v] for u, v in train.edges)


def test_diffusion_is_paired_and_reproducible():
    g = nx.cycle_graph(10)
    assert paired_cascade(g, [0, 4], 0.2, 42) == paired_cascade(g, [4, 0], 0.2, 42)
    assert paired_cascade(g, [0], 0, 42)["reach"] == 0.1
    assert paired_cascade(g, [0], 1, 42)["reach"] == 1


def test_generation_is_deterministic():
    assert build_experiment("karate", splits=2, simulations=3) == build_experiment(
        "karate", splits=2, simulations=3
    )


def test_committed_export_is_internally_consistent():
    bundle = json.loads(Path("data/experiments.json").read_text())
    assert bundle["schemaVersion"] == 1
    assert len(bundle["experiments"]) == 3
    for e in bundle["experiments"]:
        assert e["summary"]["nodes"] == len(e["nodes"])
        assert e["summary"]["edges"] == len(e["edges"])
        assert len(e["linkRuns"]) == 20
        original = {frozenset((r["source"], r["target"])) for r in e["edges"]}
        for run in e["linkRuns"]:
            assert all(frozenset(p) in original for p in run["positiveEdges"])
            assert all(frozenset(p) not in original for p in run["negativeEdges"])
            assert run["trainEdges"] + len(run["positiveEdges"]) == len(e["edges"])
        for row in e["linkComparison"]:
            scores = [
                next(s["auc"] for s in r["scores"] if s["method"] == row["method"])
                for r in e["linkRuns"]
            ]
            assert row["aucMean"] == pytest.approx(np.mean(scores))
            assert row["aucStd"] == pytest.approx(np.std(scores))
        for trial in e["influence"]:
            assert trial["mean"] == pytest.approx(np.mean(trial["reaches"]))
            assert (
                len(set(sum(trial["replay"], []))) / len(e["nodes"])
                == trial["reaches"][0]
            )
