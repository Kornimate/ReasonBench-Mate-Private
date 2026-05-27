# Clustering Metrics Summary

This document describes the metrics produced by `poe cl_visualize`, which runs
`src/visualization/clustering_metrics.py`. The pipeline executes three scripts
for each dataset in `logs/raw_calls/repeats/gpt-4.1-nano`:

1. `semantic_trajectory_metrics.py` parses raw logs, extracts task sources and
   proposal texts, embeds them, and computes semantic trajectory metrics.
2. `cluster_trajectory_metrics.py` pools proposals from all methods per case,
   clusters them into shared semantic families, and computes cluster coverage,
   discovery, redundancy, and efficiency metrics.
3. `cluster_visualizations.py` reads all saved CSV outputs and creates the kept
   visualization set: five top-level plots for each benchmark plus three
   globally selected metric plots. The three selected subfolder plots are the
   metrics where `heterogeneous_foa` and `reagents` are strongest most often
   across benchmarks.

The current command uses the deterministic TF-IDF diagnostic backend. In this
backend, semantic trajectory metrics use word and bigram TF-IDF vectors, while
cluster trajectory metrics use bounded-vocabulary TF-IDF followed by truncated
SVD and normalization before clustering.

## Notation

For a case, let:

- `f` be the source text extracted from the prompt.
- `p_t` be the `t`-th generated proposal from a method.
- `v(x)` be the embedded vector for text `x`.
- `sim(a, b) = cosine(v(a), v(b))`.
- `dist(a, b) = 1 - sim(a, b)`.
- `C` be the number of logged completions charged to the method for the case.
- `P` be the number of extracted proposals for the method and case.
- `E` be the first quartile of proposals for the method and case.
- `L` be the last quartile of proposals for the method and case.

## Semantic Trajectory Metrics

### Source alignment

For each proposal:

```text
source_alignment_t = sim(f, p_t)
```

This measures how close the proposal is to the source prompt in the embedding
space. It is a grounding proxy, not a correctness score.

Per case:

```text
source_alignment_mean = mean_t source_alignment_t
source_alignment_best = max_t source_alignment_t
best_alignment_auc = mean_t max_{i <= t} source_alignment_i
```

Reasoning:

- `source_alignment_mean` asks whether the method is consistently grounded.
- `source_alignment_best` asks whether the method ever finds a strongly
  grounded proposal.
- `best_alignment_auc` rewards finding that strong grounded proposal earlier,
  because the running best curve rises sooner.

### Semantic diversity

For a set of proposal vectors `S`:

```text
semantic_diversity(S) = mean_{a < b in S} (1 - cosine(a, b))
```

The script computes:

```text
semantic_diversity = semantic_diversity(all proposals)
early_diversity = semantic_diversity(E)
late_consensus = 1 - semantic_diversity(L)
```

Reasoning:

- High overall diversity means the method explored multiple semantic regions.
- High early diversity is useful when it happens before the method spends a lot
  of calls, because it reflects breadth near the start of search.
- High late consensus means late proposals have consolidated instead of
  continuing to scatter.

If a set contains fewer than two proposals, diversity is undefined and appears
as `NaN`.

### Grounded Exploration Efficiency

Proposal-budget version:

```text
gee_proposal_budget =
    source_alignment_best * early_diversity / ln(1 + P)
```

Full-transcript version:

```text
gee_full_transcript =
    source_alignment_best * early_diversity / ln(1 + C)
```

Explore-then-consolidate version:

```text
explore_then_consolidate_grounding =
    source_alignment_best * early_diversity * late_consensus
```

Reasoning:

- The numerator rewards methods that find grounded proposals while exploring
  distinct alternatives early.
- The logarithmic denominator charges extra sampling and tool usage with
  diminishing penalty, so a method is not punished linearly for every extra
  completion.
- `gee_full_transcript` is stricter than `gee_proposal_budget` because it counts
  all logged completions, including overhead that may not produce proposals.
- The explore-then-consolidate metric rewards a process shape: broad early
  search followed by late agreement around a grounded answer.

## Cluster Trajectory Metrics

For each case and threshold, proposals from all methods are pooled and clustered
jointly. Agglomerative clustering uses average linkage. With normalized vectors,
Euclidean distance and cosine distance are monotonic:

```text
||a - b|| = sqrt(2 * cosine_distance(a, b))
```

The script therefore uses:

```text
distance_threshold = sqrt(2 * cosine_threshold)
```

The default cosine-distance thresholds are:

```text
0.025, 0.05, 0.075, 0.10
```

### Cluster weight

For cluster `k`, let `centroid_k` be the mean vector of its proposals. The
cluster weight is:

```text
centroid_alignment_k = cosine(centroid_k, v(f))
cluster_weight_k = max(0, centroid_alignment_k)
```

Reasoning:

- A cluster represents a semantic hypothesis family.
- Clipping at zero prevents unrelated or opposed clusters from increasing
  coverage.
- Weighting by source alignment makes grounded clusters count more than
  off-source clusters.

### Coverage

For a method, let `K_m` be the unique clusters touched by its proposals, and
`K_all` be all clusters for the case:

```text
grounded_cluster_coverage =
    sum_{k in K_m} cluster_weight_k / sum_{k in K_all} cluster_weight_k
```

Reasoning:

- A method earns credit once per cluster, not once per repeated paraphrase.
- Coverage is normalized by all grounded cluster weight available in the pooled
  method set, making methods comparable within the same case and threshold.

### Discovery AUC

As proposals arrive in logged order, maintain the set of discovered clusters
`D_t`:

```text
discovery_t =
    sum_{k in D_t} cluster_weight_k / sum_{k in K_all} cluster_weight_k

grounded_cluster_discovery_auc = mean_t discovery_t
```

Reasoning:

- A method that discovers grounded clusters early has a higher curve sooner.
- The metric distinguishes efficient exploration from methods that only cover
  clusters after many proposals.

### Redundancy

```text
redundancy_rate = 1 - unique_clusters_touched / proposal_count
```

Reasoning:

- Repeated proposals in the same cluster increase redundancy.
- Lower redundancy usually means proposals are less repetitive, but it should be
  interpreted with coverage. Low redundancy with poor coverage may simply mean
  the method produced too few useful proposals.

### Effective cluster count

Let `q_k` be the fraction of the method's proposals assigned to cluster `k`:

```text
effective_cluster_count = exp(-sum_k q_k * ln(q_k))
```

Reasoning:

- This is the entropy effective number of clusters.
- It is high when proposals are spread evenly across several clusters and low
  when they collapse into one dominant cluster.

### Cluster efficiency

Full-transcript efficiency:

```text
grounded_cluster_efficiency_full =
    grounded_cluster_coverage / ln(1 + raw_completions)
```

Proposal-budget efficiency:

```text
grounded_cluster_efficiency_proposal =
    grounded_cluster_coverage / ln(1 + proposal_count)
```

Early discovery efficiency:

```text
early_grounded_cluster_efficiency_full =
    grounded_cluster_discovery_auc / ln(1 + raw_completions)
```

Reasoning:

- Coverage alone rewards breadth, but not cost.
- Efficiency divides coverage or early discovery by a logarithmic cost term.
- The full-transcript versions are the most conservative because they charge
  overhead completions, evaluator calls, and adaptive-control calls.

## Robustness Aggregation

Metrics are first computed for every clustering threshold. The robustness table
then averages method scores over thresholds:

```text
robust_metric_m = mean_threshold metric_{m, threshold}
```

For rank columns:

```text
rank_metric_{m, threshold} = rank of method m at one threshold
robust_rank_metric_m = mean_threshold rank_metric_{m, threshold}
```

The final efficiency rank is:

```text
mean_efficiency_rank =
    mean(
        rank_grounded_cluster_efficiency_full,
        rank_grounded_cluster_efficiency_proposal,
        rank_early_grounded_cluster_efficiency_full
    )
```

Reasoning:

- A single threshold can overstate or understate a method depending on cluster
  granularity.
- Averaging across thresholds checks whether the ranking is stable.
- `mean_efficiency_rank` combines full-cost coverage, proposal-budget coverage,
  and early discovery efficiency.

## Output Tables

Semantic outputs:

- `results/clustering/semantic_metrics/tfidf/<dataset>/proposals.csv`
- `results/clustering/semantic_metrics/tfidf/<dataset>/metrics_by_case.csv`
- `results/clustering/semantic_metrics/tfidf/<dataset>/metrics_by_method.csv`
- `results/clustering/semantic_metrics/tfidf/<dataset>/summary.md`

Cluster outputs:

- `results/clustering/cluster_metrics/tfidf/<dataset>/cluster_assignments.csv`
- `results/clustering/cluster_metrics/tfidf/<dataset>/clusters_by_case.csv`
- `results/clustering/cluster_metrics/tfidf/<dataset>/cluster_metrics_by_case.csv`
- `results/clustering/cluster_metrics/tfidf/<dataset>/cluster_metrics_by_method_threshold.csv`
- `results/clustering/cluster_metrics/tfidf/<dataset>/cluster_metrics_robustness.csv`
- `results/clustering/cluster_metrics/tfidf/<dataset>/summary.md`

## Plot Outputs

`cluster_visualizations.py` now writes only the kept visualization set. For each
dataset, plots are written under:

```text
results/clustering/plots/tfidf/<dataset>/
```

Each benchmark keeps eight plots total:

- Five top-level plots in `results/clustering/plots/tfidf/<dataset>/`.
- Three selected metric plots in subfolders.

### Top-level plots

- `robust_efficiency_bar.png`
- `efficiency_vs_threshold.png`
- `coverage_vs_redundancy.png`
- `semantic_map_case_auto.png`
- `semantic_map_case_<case_id>.png`

`semantic_map_case_auto.png` and `semantic_map_case_<case_id>.png` contain the
same automatically selected representative case. The case-id copy exists so the
file name records which case was selected.

Math shown by the top-level plots:

```text
robust_efficiency_bar:
    robust_grounded_cluster_efficiency_full_m =
        mean_threshold grounded_cluster_efficiency_full_{m, threshold}

efficiency_vs_threshold:
    y_{m, threshold} = grounded_cluster_efficiency_full_{m, threshold}

coverage_vs_redundancy:
    x_m = mean_threshold redundancy_rate_{m, threshold}
    y_m = mean_threshold grounded_cluster_coverage_{m, threshold}
```

The semantic map is a display-only projection:

```text
proposal texts -> TF-IDF -> TruncatedSVD -> normalization -> PCA(2D)
```

The map colors/marks proposals by method and annotates cluster centers. It is
not the metric space used for scoring; it is only an illustrative 2D view of one
representative case.

### Selected subfolder plots

Every benchmark keeps the same three selected metric plots:

- `semantic_method_metrics/source_alignment_best.png`
- `semantic_method_metrics/best_alignment_auc.png`
- `cluster_robustness_metrics/effective_cluster_count.png`

These are the three substantive metrics that most often put
`heterogeneous_foa` or `reagents` at the top or nearly at the top across
benchmarks.

Math shown by the selected subfolder plots:

```text
source_alignment_best_m =
    mean_case max_t cosine(v(f), v(p_t))

best_alignment_auc_m =
    mean_case mean_t max_{i <= t} cosine(v(f), v(p_i))

effective_cluster_count_m =
    mean_threshold mean_case exp(-sum_k q_k ln(q_k))
```

The first two plots come from `metrics_by_method.csv`; the third comes from
`cluster_metrics_robustness.csv`, after threshold robustness aggregation.

### Cross-benchmark clustering plot

The root plot directory also contains a cross-benchmark summary:

- `results/clustering/plots/tfidf/cross_benchmark_clustering_efficiency.png`
- `results/clustering/plots/tfidf/cross_benchmark_clustering_summary.csv`

For each benchmark and target method:

```text
fraction_of_benchmark_best =
    grounded_cluster_efficiency_full_target
    / max_method grounded_cluster_efficiency_full_method
```

This plot is calculated from the existing robustness CSVs. It compares
`heterogeneous_foa` and `reagents` against the best method in each benchmark.

## Interpretation Caveats

The metrics are trajectory and semantic-structure diagnostics. They should not
be read as direct task correctness scores.

- A proposal can be close to the source but still wrong.
- TF-IDF/SVD is deterministic and useful for smoke tests, but it is not a
  domain-specialized semantic model.
- Cluster coverage depends on the pooled proposal set, so it is comparative
  within a dataset/run rather than an absolute measure.
- Methods with one proposal per case can have undefined diversity-based metrics.

Use these metrics alongside task correctness, quality, cost, and latency before
making final claims about method performance.
