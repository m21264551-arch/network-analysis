# Network Analysis Toolkit

Network Analysis Toolkit is a Python project for exploring how structure shapes behaviour in social networks. It builds benchmark graphs, detects communities, ranks influential nodes, simulates information spread, and evaluates link prediction methods.

The project is designed as a compact analytics pipeline rather than a one-off notebook. It includes reusable modules, a command-line workflow, a notebook walkthrough, generated figures, tests, and GitHub Actions CI.

<p align="center">
  <img src="outputs/network_communities.png" width="700" alt="Community structure visualisation">
</p>

## What This Demonstrates

- Graph modelling with NetworkX
- Community detection and benchmark evaluation
- Centrality analysis for influence ranking
- Independent Cascade simulations
- Static link recovery with heuristic and supervised models
- Reproducible command-line analysis
- Automated tests and public repo hygiene

## Pipeline

| Step | Output |
| --- | --- |
| Build graph | SBM, LFR, or Zachary's Karate Club network |
| Detect communities | Modularity, NMI, ARI, and community counts |
| Rank nodes | Degree, betweenness, eigenvector centrality, and PageRank |
| Simulate spread | Mean reach by seed-selection strategy |
| Predict links | Held-out AUC and average precision |
| Export figures | Community, distribution, influence, and feature importance plots |

## Sample Results

These results come from a local smoke test using:

```bash
python main.py --dataset karate --output outputs_public_check
```

| Area | Example result |
| --- | --- |
| Best community modularity | Louvain, 0.4266 |
| Highest NMI | Girvan-Newman, 0.7324 |
| Top PageRank node | Node 33 |
| Best heuristic link predictor | Preferential Attachment, AUC 0.7882 |
| Supervised holdout result | Gradient Boosting, AUC 0.6181 |

Link prediction is evaluated as a static edge recovery benchmark. It does not claim to forecast future behaviour from time-series data.

## Quick Start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python main.py --dataset sbm --nodes 300
```

Other examples:

```bash
python main.py --dataset lfr --nodes 500
python main.py --dataset karate
python main.py --dataset sbm --nodes 300 --seed 7 --test-fraction 0.2
```

## Command-Line Options

```text
--dataset             Choose sbm, lfr, or karate
--nodes               Number of nodes for synthetic datasets
--output              Directory for generated figures
--seed                Random seed for reproducible runs
--test-fraction       Fraction of edges held out for link prediction
--propagation-prob    Per-edge activation probability for spread simulation
--seed-size           Number of initial seed nodes
--simulations         Number of influence simulations per strategy
```

## Tests

```bash
pytest
```

The GitHub Actions workflow runs the same test suite on every push and pull request.

## Repository Map

```text
main.py                         End-to-end CLI pipeline
src/network_builder.py          Graph construction
src/community_detection.py      Community algorithms
src/influence_analysis.py       Centrality and spread simulation
src/link_prediction.py          Link prediction features and models
src/visualisation.py            Plot generation
notebooks/analysis.ipynb        Interactive walkthrough
tests/test_pipeline.py          Regression and smoke tests
outputs/                        Generated figures
```

## Sample Figures

<p align="center">
  <img src="outputs/detection_comparison.png" width="700" alt="Algorithm comparison">
  <br><em>Community detection comparison</em>
</p>

<p align="center">
  <img src="outputs/influence_comparison.png" width="600" alt="Influence strategies">
  <br><em>Influence seed strategy comparison</em>
</p>

<p align="center">
  <img src="outputs/link_pred_importances.png" width="600" alt="Link prediction feature importances">
  <br><em>Link prediction feature importance</em>
</p>

## License

MIT
