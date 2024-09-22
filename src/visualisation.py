"""
Create consistent network analysis plots.

The plotting functions use a restrained dark theme and can save figures for
reports, notebooks, or command-line runs.
"""

from typing import Dict, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = [
    "#00D4AA",
    "#FF6B6B",
    "#4ECDC4",
    "#FFE66D",
    "#A78BFA",
    "#F97316",
    "#38BDF8",
    "#FB7185",
    "#34D399",
    "#C084FC",
    "#FACC15",
    "#2DD4BF",
]


def _apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0F172A",
            "axes.facecolor": "#0F172A",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#E2E8F0",
            "text.color": "#E2E8F0",
            "xtick.color": "#94A3B8",
            "ytick.color": "#94A3B8",
            "grid.color": "#1E293B",
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )


def plot_network_communities(
    G: nx.Graph,
    partition: Dict[int, int],
    title: str = "Community Structure",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
):
    """Draw a network coloured by community with node size scaled by degree."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    if G.number_of_nodes() == 0:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.axis("off")
        _save_if_requested(fig, save_path)
        return fig

    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

    communities = sorted(set(partition.get(node, -1) for node in G.nodes()))
    color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(communities)}
    node_colors = [color_map[partition.get(node, -1)] for node in G.nodes()]
    degree_centrality = nx.degree_centrality(G)
    node_sizes = [30 + 200 * degree_centrality[node] for node in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.08, edge_color="#475569", ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        ax=ax,
    )

    patches = [
        mpatches.Patch(color=color_map[c], label=f"Community {c}")
        for c in communities[:8]
    ]
    if patches:
        ax.legend(
            handles=patches,
            loc="upper left",
            fontsize=9,
            framealpha=0.3,
            facecolor="#1E293B",
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_detection_comparison(results: dict, save_path: Optional[str] = None):
    """Compare community detection algorithms by NMI and modularity."""
    if not results:
        raise ValueError("results must contain at least one detector result")

    _apply_style()
    names = list(results.keys())
    nmi_vals = [results[n][1]["nmi"] or 0 for n in names]
    mod_vals = [results[n][1]["modularity"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, values, metric, color in [
        (axes[0], mod_vals, "Modularity", "#00D4AA"),
        (axes[1], nmi_vals, "NMI", "#A78BFA"),
    ]:
        bars = ax.barh(names, values, color=color, alpha=0.85, height=0.5)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        _set_metric_xlim(ax, values, metric)
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=10,
                color="#E2E8F0",
            )

    plt.suptitle(
        "Community Detection: Algorithm Comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_centrality_distributions(
    centrality_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Plot distributions for each centrality metric."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["degree", "betweenness", "eigenvector", "pagerank"]
    colors = ["#00D4AA", "#FF6B6B", "#4ECDC4", "#A78BFA"]

    for ax, metric, color in zip(axes.flat, metrics, colors):
        values = centrality_df[metric].dropna()
        if len(values) < 2 or values.nunique() < 2:
            ax.hist(values, bins=1, color=color, alpha=0.5)
        else:
            values.plot.kde(ax=ax, color=color, linewidth=2)
            x, y = _kde_fill(values)
            ax.fill_between(x, y, alpha=0.2, color=color)
        ax.set_title(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Centrality Score")
        ax.set_ylabel("Density")

    plt.suptitle("Centrality Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_centrality_correlation(
    centrality_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Plot a Spearman rank correlation heatmap for centrality measures."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = centrality_df.corr(method="spearman")
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        annot_kws={"fontsize": 12, "color": "white"},
    )
    ax.set_title("Centrality Correlation (Spearman)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_influence_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Compare seed-selection strategies by mean network reach."""
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty")

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#00D4AA", "#FF6B6B", "#4ECDC4", "#A78BFA"]
    bars = ax.bar(
        comparison_df["strategy"],
        comparison_df["mean_reach"],
        yerr=comparison_df["std_reach"],
        color=colors[: len(comparison_df)],
        alpha=0.85,
        capsize=5,
        error_kw={"color": "#94A3B8", "linewidth": 1.5},
    )

    for bar, val in zip(bars, comparison_df["mean_reach"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#E2E8F0",
        )

    upper = max(
        float((comparison_df["mean_reach"] + comparison_df["std_reach"]).max()) * 1.2,
        0.05,
    )
    ax.set_ylabel("Mean Network Reach", fontsize=12)
    ax.set_title(
        "Influence Maximisation: Seed Strategy Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, upper)
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_feature_importances(
    importances: pd.Series,
    save_path: Optional[str] = None,
):
    """Plot link-prediction feature importances."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.sort_values().plot.barh(ax=ax, color="#00D4AA", alpha=0.85)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(
        "Link Prediction: Feature Importances (Gradient Boosting)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_degree_distribution(G: nx.Graph, save_path: Optional[str] = None):
    """Plot the degree distribution on log scales where valid."""
    _apply_style()
    degrees = [degree for _, degree in G.degree()]
    positive_degrees = [degree for degree in degrees if degree > 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    if positive_degrees:
        unique, counts = np.unique(positive_degrees, return_counts=True)
        probs = counts / counts.sum()
        ax.scatter(
            unique,
            probs,
            color="#00D4AA",
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "No positive-degree nodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    ax.set_xlabel("Degree (k)", fontsize=12)
    ax.set_ylabel("P(k)", fontsize=12)
    ax.set_title("Degree Distribution (log-log)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def _kde_fill(series: pd.Series) -> tuple:
    """Return x and y arrays for a KDE fill layer."""
    from scipy.stats import gaussian_kde

    values = series.dropna().astype(float)
    kde = gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), 200)
    return x, kde(x)


def _set_metric_xlim(ax, values: list, metric: str) -> None:
    """Set a readable x-axis range for bounded and unbounded metrics."""
    if metric == "NMI":
        ax.set_xlim(0, 1.05)
        return

    min_value = min(values)
    max_value = max(values)
    lower = min(0, min_value - 0.05)
    upper = max(0.05, max_value + 0.08)
    ax.set_xlim(lower, upper)


def _save_if_requested(fig, save_path: Optional[str]) -> None:
    """Save a figure when a path is provided."""
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
