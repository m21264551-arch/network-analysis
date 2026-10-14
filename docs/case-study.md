# Network Explorer

[Demo](https://network.m21264551.com/) · [Code](https://github.com/m21264551-arch/network-analysis)

On Karate Club, preferential attachment recovered held-out links better than gradient boosting: mean ROC-AUC was 0.768 versus 0.690 across 20 splits. On the weakly separated synthetic graph, the best mean was 0.500, giving no useful ranking signal in that experiment.

## Change the graph, inspect the difference

The explorer compares the observed club network with two graphs containing planted communities. Switch community methods, inspect a node, or replay simulated spread from different seed sets. Community density and agreement with reference labels are displayed separately: a partition can score well on one and poorly on the other.

## Rebuild features after removing edges

Every link-recovery split removes test edges before computing graph features. Training pairs exclude their own edge, and undirected features have the same representation when endpoints swap. One export contains the graph, partitions, trial results and exact held-out pairs, so the charts and downloads stay consistent. Spread comparisons share random propagation trials. Identical seed sets therefore produce identical reach.

## Small graphs limit the conclusion

Repeated splits reuse the same topology and are not independent datasets. Holding out edges while preserving connectivity also leaves bridges untested. The spread model describes an assumed diffusion process, not observed human behaviour. These experiments explain graph methods. They do not establish a general winner.
