# Network structure, influence and missing links

**Question.** When do node position and group separation give graph methods enough signal to be useful?

**Approach.** Use a small observed social network and two fixed stochastic block models. Export four community partitions, centralities, paired cascade experiments and 20 complete missing-edge splits in one versioned bundle. A static explorer derives its claims from that bundle.

**Result.** On Karate Club, preferential attachment averages 0.768 ROC-AUC, versus 0.690 for gradient boosting. On the weak SBM, the highest mean is 0.500. The failure is useful evidence that the methods do not recover a strong signal in that graph; it is not hidden behind a single favourable split.

**Engineering decision.** Remove row-wise CV on already-computed topology features. Rebuild the graph and all features per outer split, exclude held-out positives from every negative sampler, and sort endpoint features so an undirected pair has one representation. The export also exposes the exact test pairs for independent checking.

**Failure analysis.** A high-modularity partition may disagree with the known labels. A seed strategy can select exactly the same nodes as another; common random worlds ensure identical cascades in that case. Repeated splits of one graph are dependent and cannot establish a universally best method.

**Next limitation.** Three small fixed graphs provide controlled examples, not a broad benchmark. A temporal network dataset would be needed before claiming genuine future-link prediction.
