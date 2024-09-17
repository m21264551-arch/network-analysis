"""
visualisation.py — Publication-quality network visualisations.

All plots use a consistent dark theme with vibrant accent colours.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

# ── Style ──────────────────────────────────────
PALETTE = [
    "#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D",
    "#A78BFA", "#F97316", "#38BDF8", "#FB7185",
    "#34D399", "#C084FC", "#FACC15", "#2DD4BF",
]

def _apply_style():
    plt.rcParams.update({
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
    })


# ──────────────────────────────────────────────
# 1. Network Graph with Communities
# ──────────────────────────────────────────────

def plot_network_communities(
    G: nx.Graph,
    partition: Dict[int, int],
    title: str = "Community Structure",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
):
    """Draw the network coloured by community, node size ∝ degree."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()))

    communities = sorted(set(partition.values()))
    colour_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(communities)}
    node_colours = [colour_map[partition[n]] for n in G.nodes()]
    node_sizes = [30 + 200 * nx.degree_centrality(G)[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.08, edge_color="#475569", ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colours, node_size=node_sizes, alpha=0.85, ax=ax
    )

    patches = [mpatches.Patch(color=colour_map[c], label=f"Community {c}") for c in communities[:8]]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.3, facecolor="#1E293B")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────
# 2. Community Detection Comparison
# ──────────────────────────────────────────────

def plot_detection_comparison(
    results: dict, save_path: Optional[str] = None
):
    """Bar chart comparing community detection algorithms by NMI and Modularity."""
    _apply_style()
    names = list(results.keys())
    nmi_vals = [results[n][1]["nmi"] or 0 for n in names]
    mod_vals = [results[n][1]["modularity"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, values, metric, colour in [
        (axes[0], mod_vals, "Modularity", "#00D4AA"),
        (axes[1], nmi_vals, "NMI", "#A78BFA"),
    ]:
        bars = ax.barh(names, values, color=colour, alpha=0.85, height=0.5)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10, color="#E2E8F0")

    plt.suptitle("Community Detection — Algorithm Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────
# 3. Centrality Distributions
# ──────────────────────────────────────────────

def plot_centrality_distributions(
    centrality_df: pd.DataFrame, save_path: Optional[str] = None
):
    """KDE plots of each centrality metric."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["degree", "betweenness", "eigenvector", "pagerank"]
    colours = ["#00D4AA", "#FF6B6B", "#4ECDC4", "#A78BFA"]

    for ax, metric, colour in zip(axes.flat, metrics, colours):
        centrality_df[metric].plot.kde(ax=ax, color=colour, linewidth=2)
        ax.fill_between(
            *_kde_fill(centrality_df[metric]), alpha=0.2, color=colour
        )
        ax.set_title(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Centrality Score")
        ax.set_ylabel("Density")

    plt.suptitle("Centrality Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def _kde_fill(series):
    """Helper to get KDE x/y arrays for fill_between."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(series.dropna())
    x = np.linspace(series.min(), series.max(), 200)
    return x, kde(x)


# ──────────────────────────────────────────────
# 4. Centrality Correlation Heatmap
# ──────────────────────────────────────────────

def plot_centrality_correlation(
    centrality_df: pd.DataFrame, save_path: Optional[str] = None
):
    """Spearman rank correlation heatmap of centrality measures."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = centrality_df.corr(method="spearman")
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="viridis",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
        annot_kws={"fontsize": 12, "color": "white"},
    )
    ax.set_title("Centrality Correlation (Spearman)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────
# 5. Influence Propagation Curves
# ──────────────────────────────────────────────

def plot_influence_comparison(
    comparison_df: pd.DataFrame, save_path: Optional[str] = None
):
    """Bar chart comparing seed-selection strategies by mean network reach."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = ["#00D4AA", "#FF6B6B", "#4ECDC4", "#A78BFA"]
    bars = ax.bar(
        comparison_df["strategy"],
        comparison_df["mean_reach"],
        yerr=comparison_df["std_reach"],
        color=colours[: len(comparison_df)],
        alpha=0.85,
        capsize=5,
        error_kw={"color": "#94A3B8", "linewidth": 1.5},
    )

    for bar, val in zip(bars, comparison_df["mean_reach"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center", fontsize=12, fontweight="bold", color="#E2E8F0",
        )

    ax.set_ylabel("Mean Network Reach", fontsize=12)
    ax.set_title("Influence Maximisation — Seed Strategy Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(comparison_df["mean_reach"]) * 1.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────
# 6. Link Prediction Feature Importances
# ──────────────────────────────────────────────

def plot_feature_importances(
    importances: pd.Series, save_path: Optional[str] = None
):
    """Horizontal bar chart of link prediction feature importances."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    importances.sort_values().plot.barh(ax=ax, color="#00D4AA", alpha=0.85)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Link Prediction — Feature Importances (Gradient Boosting)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────
# 7. Degree Distribution (log-log)
# ──────────────────────────────────────────────

def plot_degree_distribution(
    G: nx.Graph, save_path: Optional[str] = None
):
    """Log-log degree distribution plot — reveals scale-free structure."""
    _apply_style()
    degrees = [d for _, d in G.degree()]
    unique, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(unique, probs, color="#00D4AA", s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (k)", fontsize=12)
    ax.set_ylabel("P(k)", fontsize=12)
    ax.set_title("Degree Distribution (log-log)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
