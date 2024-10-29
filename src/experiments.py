"""Versioned public experiments. All visible results derive from this export."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from src.community_detection import run_all_detectors
from src.influence_analysis import compute_centralities
from src.link_prediction import (
    train_test_split_edges,
    evaluate_heuristics,
    train_link_predictor,
)
from src.network_builder import (
    load_karate_club,
    generate_stochastic_block_model,
    get_network_summary,
)

DATASETS = {
    "karate": {
        "name": "Karate Club",
        "description": "34 members, with friendship ties observed by Wayne Zachary. The reference labels describe the two groups after the club split.",
        "source": "https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html",
    },
    "sbm-strong": {
        "name": "Strongly separated groups",
        "description": "A synthetic graph with four planted groups of 20 nodes. Within-group edge probability 0.30; between-group probability 0.015.",
        "p_intra": 0.30,
        "p_inter": 0.015,
    },
    "sbm-weak": {
        "name": "Weakly separated groups",
        "description": "The same four planted groups, but more cross-group links: within-group probability 0.18; between-group probability 0.08.",
        "p_intra": 0.18,
        "p_inter": 0.08,
    },
}


def paired_cascade(graph, seeds, probability, trial):
    """Independent cascade via common directed live-edge worlds for paired trials."""
    rng = np.random.default_rng(trial)
    active_edges = {
        (u, v)
        for u, v in sorted(graph.to_directed().edges())
        if rng.random() < probability
    }
    active, frontier = set(seeds), set(seeds)
    history = [sorted(active)]
    while frontier:
        following = {
            v for u in frontier for v in graph.neighbors(u) if (u, v) in active_edges
        } - active
        if not following:
            break
        active |= following
        history.append(sorted(following))
        frontier = following
    return {"reach": len(active) / max(1, len(graph)), "history": history}


def build_experiment(key, splits=20, simulations=100):
    info = DATASETS[key]
    graph = (
        load_karate_club()
        if key == "karate"
        else generate_stochastic_block_model(
            [20] * 4, info["p_intra"], info["p_inter"], seed=42
        )
    )
    # One explicit unweighted policy across all algorithms and simulations.
    for _, _, data in graph.edges(data=True):
        data.clear()
    centrality = compute_centralities(graph)
    labels = sorted(
        set(nx.get_node_attributes(graph, "ground_truth").values()), key=str
    )
    reference = {
        node: labels.index(graph.nodes[node]["ground_truth"]) for node in graph
    }
    detectors = run_all_detectors(graph, k=len(labels))
    if len(detectors) != 4:
        raise RuntimeError("All four detectors must complete before publication")
    position = nx.spring_layout(graph, seed=42)
    nodes = [
        {
            "id": int(node),
            "x": float(position[node][0]),
            "y": float(position[node][1]),
            "reference": reference[node],
            "degree_count": int(graph.degree(node)),
            **{metric: float(centrality.loc[node, metric]) for metric in centrality},
        }
        for node in sorted(graph)
    ]
    edges = [{"source": int(u), "target": int(v)} for u, v in sorted(graph.edges())]
    communities = [
        {"method": method, "partition": partition, **metrics}
        for method, (partition, metrics) in detectors.items()
    ]
    influence = []
    for count in [1, 3, 5]:
        strategies = {
            name: centrality.sort_values([metric], ascending=False, kind="stable")
            .head(count)
            .index.tolist()
            for name, metric in [
                ("PageRank", "pagerank"),
                ("Degree", "degree"),
                ("Betweenness", "betweenness"),
            ]
        }
        # Random strategy is a single seeded choice; trials vary diffusion only.
        strategies["Random"] = (
            np.random.default_rng(42)
            .choice(sorted(graph), count, replace=False)
            .tolist()
        )
        for probability in [0.05, 0.1, 0.2]:
            for strategy, seeds in strategies.items():
                trials = [
                    paired_cascade(graph, seeds, probability, trial)
                    for trial in range(simulations)
                ]
                reaches = [row["reach"] for row in trials]
                influence.append(
                    {
                        "strategy": strategy,
                        "seedCount": count,
                        "probability": probability,
                        "seeds": [int(s) for s in seeds],
                        "reaches": reaches,
                        "mean": float(np.mean(reaches)),
                        "std": float(np.std(reaches)),
                        "replay": trials[0]["history"],
                    }
                )
    link_runs = []
    for seed in range(splits):
        train, positives, negatives = train_test_split_edges(
            graph, test_fraction=0.15, seed=seed
        )
        heuristic = evaluate_heuristics(train, positives, negatives)
        model = train_link_predictor(train, positives, negatives, seed=seed)
        scores = [
            {"method": row["method"], "auc": row["AUC-ROC"], "ap": row["Avg Precision"]}
            for row in heuristic.to_dict("records")
        ]
        scores.append(
            {
                "method": "Gradient Boosting",
                "auc": model["test_auc"],
                "ap": model["test_average_precision"],
            }
        )
        link_runs.append(
            {
                "seed": seed,
                "trainEdges": train.number_of_edges(),
                "positiveEdges": positives,
                "negativeEdges": negatives,
                "scores": scores,
            }
        )
    comparison = []
    for method in [row["method"] for row in link_runs[0]["scores"]]:
        scores = [
            next(s for s in run["scores"] if s["method"] == method) for run in link_runs
        ]
        comparison.append(
            {
                "method": method,
                "aucMean": float(np.mean([s["auc"] for s in scores])),
                "aucStd": float(np.std([s["auc"] for s in scores])),
                "apMean": float(np.mean([s["ap"] for s in scores])),
                "aucValues": [s["auc"] for s in scores],
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "id": key,
        **info,
        "datasetChecksum": fingerprint,
        "summary": get_network_summary(graph),
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "influence": influence,
        "linkRuns": link_runs,
        "linkComparison": comparison,
    }


def export_dashboard(path, splits=20, simulations=100):
    root = Path(__file__).resolve().parents[1]
    source_checksum = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((root / "src").glob("*.py")))
    ).hexdigest()
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "codeRevision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "sourceChecksum": source_checksum,
        "parameters": {
            "graphSeed": 42,
            "layoutSeed": 42,
            "outerSplitSeeds": list(range(splits)),
            "testFraction": 0.15,
            "simulations": simulations,
            "weightPolicy": "unweighted",
            "negativeSampling": "balanced true non-edges",
            "diffusion": "paired directed live-edge worlds; fixed seed sets",
        },
        "experiments": [build_experiment(key, splits, simulations) for key in DATASETS],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f'Exported {len(payload["experiments"])} experiments to {path}')
    return payload
