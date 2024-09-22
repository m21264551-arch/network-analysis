"""
End-to-end NetDetect pipeline.

Runs graph construction, community detection, centrality analysis, influence
propagation, link prediction, and plot generation.

Usage
-----
    python main.py [--dataset {sbm,lfr,karate}] [--nodes 300] [--output outputs]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.community_detection import run_all_detectors
from src.influence_analysis import compare_seed_strategies, compute_centralities
from src.link_prediction import (
    evaluate_heuristics,
    train_link_predictor,
    train_test_split_edges,
)
from src.network_builder import (
    generate_lfr_benchmark,
    generate_stochastic_block_model,
    get_network_summary,
    load_karate_club,
)
from src.visualisation import (
    plot_centrality_correlation,
    plot_centrality_distributions,
    plot_degree_distribution,
    plot_detection_comparison,
    plot_feature_importances,
    plot_influence_comparison,
    plot_network_communities,
)


def main(dataset: str = "sbm", n_nodes: int = 300, output_dir: str = "outputs"):
    """Run the full analysis pipeline and return the main result objects."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NetDetect Social Network Analysis Pipeline")
    print("=" * 60)

    node_label = f", n={n_nodes}" if dataset != "karate" else ""
    print(f"\n[1/6] Building network ({dataset}{node_label})...")
    G = _build_graph(dataset, n_nodes)

    summary = get_network_summary(G)
    for key, value in summary.items():
        print(f"  {key:20s}: {value}")

    plot_degree_distribution(G, save_path=str(out / "degree_distribution.png"))
    print("  [ok] Degree distribution saved")

    print("\n[2/6] Running community detection algorithms...")
    k = _community_count_for_detection(G)
    results = run_all_detectors(G, k=k)
    if not results:
        raise RuntimeError("No community detection algorithms completed successfully")

    for name, (_, metrics) in results.items():
        print(
            f"  {name:25s} | Modularity={metrics['modularity']:.4f}  "
            f"NMI={metrics['nmi']}  Communities={metrics['num_communities']}"
        )

    best_name = max(results, key=lambda n: results[n][1]["modularity"])
    best_partition = results[best_name][0]

    plot_network_communities(
        G,
        best_partition,
        title=f"Communities ({best_name})",
        save_path=str(out / "network_communities.png"),
    )
    plot_detection_comparison(results, save_path=str(out / "detection_comparison.png"))
    print("  [ok] Community plots saved")

    print("\n[3/6] Computing centrality measures...")
    centrality_df = compute_centralities(G)
    print("  Top 5 by PageRank:")
    print(centrality_df.head().to_string(float_format="{:.4f}".format))

    plot_centrality_distributions(
        centrality_df,
        save_path=str(out / "centrality_distributions.png"),
    )
    plot_centrality_correlation(
        centrality_df,
        save_path=str(out / "centrality_correlation.png"),
    )
    print("  [ok] Centrality plots saved")

    print("\n[4/6] Simulating influence propagation...")
    comparison = compare_seed_strategies(
        G,
        centrality_df,
        seed_size=5,
        propagation_prob=0.1,
    )
    print(comparison.to_string(index=False))

    plot_influence_comparison(comparison, save_path=str(out / "influence_comparison.png"))
    print("  [ok] Influence plot saved")

    print("\n[5/6] Running link prediction experiments...")
    G_train, test_pos, test_neg = train_test_split_edges(G, test_fraction=0.15)
    print(
        f"  Train edges: {G_train.number_of_edges()}  |  "
        f"Test +: {len(test_pos)}  |  Test -: {len(test_neg)}"
    )

    heuristic_results = evaluate_heuristics(G_train, test_pos, test_neg)
    print("\n  Heuristic methods:")
    print(heuristic_results.to_string(index=False))

    ml_results = train_link_predictor(G_train, test_pos, test_neg)
    cv_mean = ml_results["cv_mean_auc"]
    cv_std = ml_results["cv_std_auc"]
    cv_text = "not available" if cv_mean is None else f"{cv_mean:.4f} +/- {cv_std:.4f}"
    print(
        "\n  Gradient Boosting"
        f"  |  Holdout AUC = {ml_results['test_auc']:.4f}"
        f"  |  Avg Precision = {ml_results['test_average_precision']:.4f}"
        f"  |  Train CV AUC = {cv_text}"
    )

    plot_feature_importances(
        ml_results["feature_importances"],
        save_path=str(out / "link_pred_importances.png"),
    )
    print("  [ok] Link prediction plot saved")

    print("\n[6/6] Pipeline complete")
    print(f"  All outputs saved to: {out.resolve()}")
    print("=" * 60)

    return {
        "summary": summary,
        "community_results": {n: m for n, (_, m) in results.items()},
        "top_influencers": centrality_df.head(10),
        "influence_comparison": comparison,
        "heuristic_link_pred": heuristic_results,
        "ml_link_pred_auc": ml_results["test_auc"],
    }


def _build_graph(dataset: str, n_nodes: int):
    """Build a graph for the selected dataset."""
    if dataset == "sbm":
        return generate_stochastic_block_model(
            sizes=_balanced_community_sizes(n_nodes, 5),
            p_intra=0.25,
            p_inter=0.01,
        )
    if dataset == "lfr":
        return generate_lfr_benchmark(n=n_nodes)
    if dataset == "karate":
        return load_karate_club()
    raise ValueError(f"Unknown dataset: {dataset}")


def _balanced_community_sizes(n_nodes: int, num_communities: int = 5) -> list[int]:
    """Split ``n_nodes`` across communities while preserving the exact total."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    num_communities = min(num_communities, n_nodes)
    base_size, remainder = divmod(n_nodes, num_communities)
    return [
        base_size + (1 if index < remainder else 0)
        for index in range(num_communities)
    ]


def _community_count_for_detection(G) -> int:
    """Pick a valid community count for algorithms that require k."""
    ground_truth = {
        value
        for value in dict(G.nodes(data="ground_truth")).values()
        if value is not None
    }
    if ground_truth:
        return min(len(ground_truth), G.number_of_nodes())
    return min(3, G.number_of_nodes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NetDetect Social Network Analysis Pipeline",
    )
    parser.add_argument("--dataset", choices=["sbm", "lfr", "karate"], default="sbm")
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()
    main(dataset=args.dataset, n_nodes=args.nodes, output_dir=args.output)
