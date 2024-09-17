# NetDetect — Social Network Analysis & Influence Propagation

A graph analytics pipeline that detects communities, identifies influential nodes, simulates information spread, and predicts future connections in social networks.

<p align="center">
  <img src="outputs/network_communities.png" width="700" alt="Community structure visualisation">
</p>

## Key Findings

| Component | Result |
|---|---|
| **Community Detection** | Louvain achieves highest modularity with near-perfect NMI on planted benchmarks |
| **Centrality Analysis** | PageRank and Eigenvector centrality are strongly correlated (Spearman ρ > 0.9) |
| **Influence Maximisation** | Betweenness-based seed selection reaches ~3× more of the network than random |
| **Link Prediction** | Supervised Gradient Boosting (AUC ~0.95+) outperforms all heuristic baselines |

## Project Structure

```
netdetect/
├── main.py                     # End-to-end CLI pipeline
├── requirements.txt
├── README.md
├── notebooks/
│   └── analysis.ipynb          # Interactive walkthrough
├── src/
│   ├── network_builder.py      # Graph construction (SBM, LFR, real datasets)
│   ├── community_detection.py  # Louvain, LPA, Girvan-Newman, Spectral
│   ├── influence_analysis.py   # Centrality + Independent Cascade simulation
│   ├── link_prediction.py      # Heuristic + ML-based link prediction
│   └── visualisation.py        # Publication-quality plots
└── outputs/                    # Generated figures
```

## Techniques & Algorithms

### Community Detection
- **Louvain** — Greedy modularity optimisation, O(n log n)
- **Label Propagation** — Near-linear, parameter-free
- **Girvan-Newman** — Edge-betweenness removal, interpretable
- **Spectral Clustering** — Graph Laplacian eigenvectors + k-means

### Centrality & Influence
- Degree, Betweenness, Eigenvector, and PageRank centrality
- **Independent Cascade (IC)** model for influence propagation
- Seed strategy comparison for influence maximisation

### Link Prediction
- Heuristics: Common Neighbours, Jaccard, Adamic-Adar, Preferential Attachment
- Supervised ML: Gradient Boosting on hand-crafted node-pair features
- Evaluation: AUC-ROC, Average Precision, 5-fold CV

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py --dataset sbm --nodes 300

# Or use a different dataset
python main.py --dataset lfr --nodes 500
python main.py --dataset karate
```

### Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

## Tech Stack

- **Graph Analysis**: NetworkX
- **Machine Learning**: scikit-learn (Gradient Boosting, Spectral Clustering, evaluation metrics)
- **Visualisation**: matplotlib, seaborn
- **Data**: numpy, pandas, scipy

## Sample Outputs

<p align="center">
  <img src="outputs/detection_comparison.png" width="700" alt="Algorithm comparison">
  <br><em>Community detection algorithm comparison</em>
</p>

<p align="center">
  <img src="outputs/influence_comparison.png" width="600" alt="Influence strategies">
  <br><em>Seed selection strategy comparison for influence maximisation</em>
</p>

<p align="center">
  <img src="outputs/link_pred_importances.png" width="600" alt="Feature importances">
  <br><em>Link prediction feature importances</em>
</p>

## Extending This Project

- **Temporal networks** — analyse how communities evolve over time
- **Graph Neural Networks** — GCN / GraphSAGE for node classification
- **Real-world data** — Twitter follower graphs, citation networks, Reddit interactions
- **Overlapping communities** — BigCLAM, CESNA algorithms

## License

MIT
