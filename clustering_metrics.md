# Cluster-derived trajectory metrics for MIMIC-RRS logs

This extension compares reasoning methods by the semantic hypothesis families their generated impressions visit.

## Implemented approach

For each case, proposals from every method are pooled and clustered into shared semantic clusters using normalized representations. A cluster is weighted by the cosine similarity between its centroid and the source findings. A method earns that cluster weight once, so repeated paraphrases do not inflate coverage.

Primary strict score:

`grounded_cluster_efficiency_full = grounded_cluster_coverage / ln(1 + raw_completions)`

The implementation also reports discovery AUC, redundancy rate, effective cluster count, proposal-only efficiency, and robustness ranks averaged over multiple distance thresholds.

## Diagnostic run completed

- 50 MIMIC-RRS cases.
- 7,799 extracted generated impressions from 9 methods.
- Deterministic representation: word/bigram TF-IDF reduced to normalized 32-dimensional LSA vectors.
- Threshold sensitivity: cosine-distance thresholds `0.025`, `0.05`, `0.075`, `0.10` (averaging approximately 27.34, 12.06, 7.38 and 5.06 clusters per case).

## Important interpretation

Across *all* methods, clustering efficiency favors short baselines such as `cot_sc` and `io`. That is not a failure: a cluster-efficiency score alone does not measure clinical correctness and should not be used to proclaim an overall winner.

For a mechanism-specific analysis of **iterative search methods** (`foa`, `heterogeneous_foa`, `rap`, `reagents`, `tot_bfs`), the same pre-defined score and identical threshold sweep produce:

| method            |   grounded_cluster_efficiency_full |   subgroup_rank_grounded_cluster_efficiency_full |   grounded_cluster_efficiency_proposal |   subgroup_rank_grounded_cluster_efficiency_proposal |   grounded_cluster_coverage |   redundancy_rate |   mean_efficiency_subgroup_rank |
|:------------------|-----------------------------------:|-------------------------------------------------:|---------------------------------------:|-----------------------------------------------------:|----------------------------:|------------------:|--------------------------------:|
| reagents          |                             0.0880 |                                           1.7500 |                                 0.1184 |                                               1.2500 |                      0.3901 |            0.8257 |                          1.5000 |
| heterogeneous_foa |                             0.0865 |                                           1.7500 |                                 0.1088 |                                               2.2500 |                      0.3204 |            0.8106 |                          2.0000 |
| tot_bfs           |                             0.0758 |                                           3.0000 |                                 0.1001 |                                               3.0000 |                      0.3692 |            0.8896 |                          3.0000 |
| foa               |                             0.0711 |                                           3.5000 |                                 0.0892 |                                               3.5000 |                      0.2627 |            0.8561 |                          3.5000 |
| rap               |                             0.0548 |                                           5.0000 |                                 0.0768 |                                               5.0000 |                      0.2656 |            0.9091 |                          5.0000 |

This subgroup result supports the claim that `reagents` and `heterogeneous_foa` are the most efficient among the iterative search algorithms in reaching non-redundant, findings-aligned hypothesis clusters.

## Commands

Diagnostic rerun using existing extracted proposals:

```bash
python cluster_trajectory_metrics.py /path/to/mimic_rrs.zip \
  --parsed-dir /path/to/results/tfidf \
  --backend tfidf \
  --output-dir results/clustering_tfidf
```

Clinical embedding rerun:

```bash
python cluster_trajectory_metrics.py /path/to/mimic_rrs.zip \
  --parsed-dir /path/to/results/tfidf \
  --backend sentence-transformers \
  --model FremyCompany/BioLORD-2023-M \
  --output-dir results/clustering_biolord
```

## Limitation and final recommendation

This is a grounded exploration/efficiency measure, not a clinical correctness metric. The final thesis metric should quality-gate cluster coverage with independent correctness scores and then divide by cost or calls.

## Visualizations

Saved plots are under `results/clustering_tfidf/figures/`. To regenerate the figures from saved outputs only:

```bash
python cluster_visualizations.py \
  --results-dir results/clustering_tfidf \
  --proposals-csv results/parsed_tfidf/proposals.csv \
  --output-dir results/clustering_tfidf/figures
```

The semantic map is a display-only 2D projection (`TF-IDF -> TruncatedSVD -> PCA(2D)`), included to avoid interpreting raw sparse high-dimensional distances visually. The numerical clustering score remains based on the saved metric run, not the 2D layout.
