# Clustering Metrics Summary

Using deterministic TF-IDF backend. In this
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