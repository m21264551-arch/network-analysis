"""
main.py — End-to-end NetDetect pipeline.

Runs the full analysis: graph construction → community detection →
centrality analysis → influence propagation → link prediction → visualisations.

Usage
-----
    python main.py [--dataset {sbm,lfr,karate}] [--nodes 300] [--output outputs]
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.network_builder import (
    generate_stochastic_block_model,
    generate_lfr_benchmark,
    load_karate_club,
    get_network_summary,
)
from src.community_detection import run_all_detectors
from src.influence_analysis import compute_centralities, compare_seed_strategies
from src.link_prediction import train_test_split_edges, evaluate_heuristics, train_link_predictor
from src.visualisation import (
    plot_network_communities,
    plot_detection_comparison,
    plot_centrality_distributions,
    plot_centrality_correlation,
    plot_influence_comparison,
    plot_feature_importances,
    plot_degree_distribution,
)

import matplotlib
matplotlib.use("Agg")


def main(dataset: str = "sbm", n_nodes: int = 300, output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # ── 1. Build Network ──────────────────────────
    print("=" * 60)
    print("  NetDetect — Social Network Analysis Pipeline")
    print("=" * 60)
    print(f"\n[1/6] Building network ({dataset}, n={n_nodes})...")

    if dataset == "sbm":
        G = generate_stochastic_block_model(
            sizes=[n_nodes // 5] * 5, p_intra=0.25, p_inter=0.01
        )
    elif dataset == "lfr":
        G = generate_lfr_benchmark(n=n_nodes)
    elif dataset == "karate":
        G = load_karate_club()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    summary = get_network_summary(G)
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    plot_degree_distribution(G, save_path=str(out / "degree_distribution.png"))
    print("  ✓ Degree distribution saved")

    # ── 2. Community Detection ────────────────────
    print(f"\n[2/6] Running community detection algorithms...")
    n_communities = len(set(
        v for v in dict(G.nodes(data="ground_truth")).values() if v is not None
    ))
    k = max(n_communities, 3)
    results = run_all_detectors(G, k=k)

    for name, (partition, metrics) in results.items():
        print(f"  {name:25s} | Modularity={metrics['modularity']:.4f}  "
              f"NMI={metrics['nmi']}  Communities={metrics['num_communities']}")

    # Use Louvain partition for visualisation
    best_name = max(results, key=lambda n: results[n][1]["modularity"])
    best_partition = results[best_name][0]

    plot_network_communities(G, best_partition, title=f"Communities ({best_name})",
                             save_path=str(out / "network_communities.png"))
    plot_detection_comparison(results, save_path=str(out / "detection_comparison.png"))
    print("  ✓ Community plots saved")

    # ── 3. Centrality Analysis ────────────────────
    print(f"\n[3/6] Computing centrality measures...")
    centrality_df = compute_centralities(G)
    print(f"  Top 5 by PageRank:")
    print(centrality_df.head().to_string(float_format="{:.4f}".format))

    plot_centrality_distributions(centrality_df, save_path=str(out / "centrality_distributions.png"))
    plot_centrality_correlation(centrality_df, save_path=str(out / "centrality_correlation.png"))
    print("  ✓ Centrality plots saved")

    # ── 4. Influence Propagation ──────────────────
    print(f"\n[4/6] Simulating influence propagation (Independent Cascade)...")
    comparison = compare_seed_strategies(G, centrality_df, seed_size=5, propagation_prob=0.1)
    print(comparison.to_string(index=False))

    plot_influence_comparison(comparison, save_path=str(out / "influence_comparison.png"))
    print("  ✓ Influence plot saved")

    # ── 5. Link Prediction ────────────────────────
    print(f"\n[5/6] Running link prediction experiments...")
    G_train, test_pos, test_neg = train_test_split_edges(G, test_fraction=0.15)
    print(f"  Train edges: {G_train.number_of_edges()}  |  "
          f"Test +: {len(test_pos)}  |  Test -: {len(test_neg)}")

    heuristic_results = evaluate_heuristics(G_train, test_pos, test_neg)
    print("\n  Heuristic Methods:")
    print(heuristic_results.to_string(index=False))

    ml_results = train_link_predictor(G_train, test_pos, test_neg)
    print(f"\n  Gradient Boosting  |  CV AUC = {ml_results['mean_auc']:.4f} ± {ml_results['std_auc']:.4f}")

    plot_feature_importances(ml_results["feature_importances"],
                             save_path=str(out / "link_pred_importances.png"))
    print("  ✓ Link prediction plot saved")

    # ── 6. Summary ────────────────────────────────
    print(f"\n[6/6] Pipeline complete!")
    print(f"  All outputs saved to: {out.resolve()}")
    print("=" * 60)

    return {
        "summary": summary,
        "community_results": {n: m for n, (_, m) in results.items()},
        "top_influencers": centrality_df.head(10),
        "influence_comparison": comparison,
        "heuristic_link_pred": heuristic_results,
        "ml_link_pred_auc": ml_results["mean_auc"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetDetect — Network Analysis Pipeline")
    parser.add_argument("--dataset", choices=["sbm", "lfr", "karate"], default="sbm")
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()
    main(dataset=args.dataset, n_nodes=args.nodes, output_dir=args.output)
