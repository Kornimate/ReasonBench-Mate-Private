# ReasonBench Log-Metric Evaluation Summary

## Scope and evaluation integrity

This analysis summarizes the logs generated for **9 methods** on **11 benchmarks**, all using `gpt-4.1-nano`. The denominator is derived from the recorded `Quality: Correct: [...]` score arrays rather than from filenames. This is necessary because `matharena` records only **15** evaluated samples per method, whereas each of the other ten benchmarks records **50**:

\[
N_{\text{per method}} = 10 \cdot 50 + 15 = 515.
\]

The solved thresholds follow the runtime benchmark environments:

| Benchmark class | Solved threshold |
|---|---:|
| Binary-score benchmarks | \(q \ge 1.0\) |
| `mimic_rrs` | \(q \ge 4.0\) |
| `mtsamples_procedures` | \(q \ge 3.8\) |

The two medical tasks report grader scores on a five-point scale. They are normalized before cross-benchmark quality aggregation; otherwise their larger numeric scale would dominate the binary benchmarks.

---

## Main result

`reagents` is the strongest method on overall effectiveness: it has the best macro normalized quality, solves the most instances, and ranks first under the declared performance-priority composite score. `heterogeneous_foa` is close behind in quality and solved count, while using substantially fewer resources.

| Comparison | Result |
|---|---:|
| Macro normalized quality: `reagents` | **0.6048** |
| Macro normalized quality: `heterogeneous_foa` | **0.5921** |
| Quality difference (`reagents` − `heterogeneous_foa`) | **0.0127** |
| Solved instances | **260 vs 254** out of 515 |
| Solved-instance difference | **6** |
| Monetary-cost reduction of `heterogeneous_foa` relative to `reagents` | **27.79%** |
| Call-count reduction of `heterogeneous_foa` relative to `reagents` | **26.34%** |
| Paired benchmark-stratified bootstrap 95% CI for the quality difference | **[-0.0098, 0.0349]** |

The bootstrap interval contains zero. Therefore, the present logs support describing `reagents` as the observed quality leader and `heterogeneous_foa` as a near-quality-equivalent, lower-cost alternative; they do **not** establish a statistically clear quality advantage of `reagents` over `heterogeneous_foa`.

---

## Overall method summary

| Method            |   Attempts |   Solved | Micro solve rate   |   Macro quality |   Macro solve rate |   Harmonic effectiveness | Cost    | Calls   |   Composite rank | Pareto-optimal   |
|:------------------|-----------:|---------:|:-------------------|----------------:|-------------------:|-------------------------:|:--------|:--------|-----------------:|:-----------------|
| reagents          |        515 |      260 | 50.49%             |          0.6048 |             0.4727 |                   0.5307 | $2.9911 | 24,683  |             2.8  | Yes              |
| heterogeneous_foa |        515 |      254 | 49.32%             |          0.5921 |             0.4618 |                   0.5189 | $2.1598 | 18,181  |             2.9  | Yes              |
| tot_bfs           |        515 |      248 | 48.16%             |          0.5457 |             0.4509 |                   0.4938 | $4.6972 | 41,983  |             4.65 | No               |
| foa               |        515 |      190 | 36.89%             |          0.5096 |             0.3455 |                   0.4118 | $2.3563 | 21,199  |             4.6  | No               |
| io                |        515 |      185 | 35.92%             |          0.4059 |             0.3364 |                   0.3679 | $0.0725 | 565     |             3.8  | Yes              |
| rap               |        515 |      159 | 30.87%             |          0.3695 |             0.2891 |                   0.3244 | $4.4962 | 73,936  |             6.75 | No               |
| cot               |        515 |      114 | 22.14%             |          0.3504 |             0.2073 |                   0.2605 | $0.1277 | 565     |             5.65 | No               |
| cot_sc            |        515 |      105 | 20.39%             |          0.3295 |             0.1909 |                   0.2418 | $0.5725 | 5,250   |             6.95 | No               |
| react             |        515 |      143 | 27.77%             |          0.3258 |             0.26   |                   0.2892 | $0.6122 | 4,133   |             6.75 | No               |

---

## Metric definitions

Let:

- \(m\) denote a method;
- \(d \in D\) denote a benchmark, with \(|D| = 11\);
- \(i\) denote an evaluated sample within a benchmark;
- \(q_{m,d,i}\) denote the recorded raw quality score;
- \(M_d\) denote the maximum score scale used for normalization, with \(M_d=5\) for the medical tasks and \(M_d=1\) otherwise;
- \(\tau_d\) denote the solved threshold for benchmark \(d\);
- \(n_d\) denote the number of evaluated samples in benchmark \(d\);
- \(C_m\) and \(A_m\) denote total dollar cost and total model calls for method \(m\).

### 1. Normalized per-sample quality

\[
\tilde{q}_{m,d,i} = \frac{q_{m,d,i}}{M_d}.
\]

This maps all benchmark quality values to a comparable \([0,1]\) range.

### 2. Per-benchmark normalized quality

\[
Q_{m,d} = \frac{1}{n_d} \sum_{i=1}^{n_d} \tilde{q}_{m,d,i}.
\]

### 3. Macro normalized quality

\[
Q^{\mathrm{macro}}_m =
\frac{1}{|D|} \sum_{d \in D} Q_{m,d}.
\]

Every benchmark contributes equally, so `matharena` does not become less important simply because it has fewer evaluated samples. This is the primary overall quality measure.

### 4. Micro normalized quality

\[
Q^{\mathrm{micro}}_m =
\frac{\sum_{d \in D} \sum_{i=1}^{n_d} \tilde{q}_{m,d,i}}
{\sum_{d \in D} n_d}.
\]

Every evaluated instance contributes equally; consequently, benchmarks with 50 samples contribute more than `matharena`, which has 15.

### 5. Solved instances and solve rates

\[
S_{m,d} =
\sum_{i=1}^{n_d}
\mathbf{1}\left[q_{m,d,i} \geq \tau_d\right],
\qquad
SR_{m,d} = \frac{S_{m,d}}{n_d}.
\]

Total solved instances and the micro solve rate are:

\[
S_m = \sum_{d \in D} S_{m,d},
\qquad
SR^{\mathrm{micro}}_m =
\frac{S_m}{\sum_{d \in D}n_d}.
\]

Macro solve rate gives each benchmark equal weight:

\[
SR^{\mathrm{macro}}_m =
\frac{1}{|D|} \sum_{d \in D} SR_{m,d}.
\]

### 6. Harmonic effectiveness

To require strength in both graded quality and thresholded solving, the script computes:

\[
H_m =
\frac{2 Q^{\mathrm{macro}}_m SR^{\mathrm{macro}}_m}
{Q^{\mathrm{macro}}_m + SR^{\mathrm{macro}}_m}.
\]

This is a project-defined summary metric inspired by harmonic-mean measures such as the F-measure: a method scores highly only when both components are high. It is not presented as a standard ReasonBench metric.

### 7. Resource-efficiency metrics

\[
\mathrm{CostPerSolved}_m = \frac{C_m}{S_m},
\qquad
\mathrm{CallsPerSolved}_m = \frac{A_m}{S_m},
\]

\[
\mathrm{QualityPerDollar}_m =
\frac{Q^{\mathrm{macro}}_m}{C_m},
\qquad
\mathrm{QualityPer1000Calls}_m =
\frac{Q^{\mathrm{macro}}_m}{A_m / 1000},
\]

\[
\mathrm{SolvedPerDollar}_m =
\frac{S_m}{C_m},
\qquad
\mathrm{SolvedPer1000Calls}_m =
\frac{S_m}{A_m / 1000}.
\]

These resource-normalized metrics favor cheap single-pass methods such as `io`, so they should be interpreted alongside effectiveness rather than as standalone quality rankings.

### 8. Performance-priority composite rank

The script ranks each method on macro quality, total solved instances, total cost, and total calls, and combines the ranks as:

\[
R^{\mathrm{composite}}_m =
0.40R^Q_m +
0.30R^S_m +
0.15R^C_m +
0.15R^A_m,
\]

where quality and solved ranks are descending-performance ranks, while cost and call ranks prefer smaller values. The weighting gives **70%** of the aggregate importance to effectiveness and **30%** to resource use.

| Rank | Method | Composite-rank value |
|---:|---|---:|
| 1 | `reagents` | 2.80 |
| 2 | `heterogeneous_foa` | 2.90 |

This metric must be reported with its explicit weights; it is a declared decision rule, not an objective universal ordering.

### 9. Four-dimensional Pareto frontier

Method \(a\) dominates method \(b\) when it is at least as good on both effectiveness dimensions and no more expensive on both resource dimensions, with a strict improvement in at least one dimension:

\[
Q^{\mathrm{macro}}_a \ge Q^{\mathrm{macro}}_b,
\quad
S_a \ge S_b,
\quad
C_a \le C_b,
\quad
A_a \le A_b.
\]

The Pareto-optimal methods in these logs are:

| Method | Role in the frontier |
|---|---|
| `reagents` | Highest observed effectiveness |
| `heterogeneous_foa` | High-effectiveness, lower-resource trade-off |
| `io` | Very low-resource, lower-quality trade-off |

### 10. Paired benchmark-stratified bootstrap uncertainty

For the top two methods, a paired difference is computed for every shared sample in each benchmark:

\[
\Delta_{d,i} =
\tilde{q}_{\text{reagents},d,i}
-
\tilde{q}_{\text{heterogeneous\_foa},d,i}.
\]

Within each benchmark, paired differences are resampled with replacement. The mean difference is computed per benchmark and then averaged across benchmarks, maintaining macro benchmark weighting. With 10,000 bootstrap iterations:

\[
\widehat{\Delta} = 0.0127,
\qquad
CI_{95\%} = [-0.0098, 0.0349].
\]

Because the interval overlaps zero, the current experiment does not demonstrate a reliable quality separation between the top two methods.

---

## Benchmark-level difficulty for `reagents` and `heterogeneous_foa`

The benchmarks below are ordered by the mean normalized quality of the two focus methods, from hardest to easiest.

|   Difficulty rank | Benchmark            |   Attempts/method |   ReAgents quality |   Het-FoA quality | ReAgents solved   | Het-FoA solved   | Quality leader    |
|------------------:|:---------------------|------------------:|-------------------:|------------------:|:------------------|:-----------------|:------------------|
|                 1 | matharena            |                15 |             0      |            0      | 0.0%              | 0.0%             | tie               |
|                 2 | game24               |                50 |             0.24   |            0.34   | 24.0%             | 34.0%            | heterogeneous_foa |
|                 3 | hotpotqa             |                50 |             0.3    |            0.34   | 30.0%             | 34.0%            | heterogeneous_foa |
|                 4 | scibench             |                50 |             0.32   |            0.36   | 32.0%             | 36.0%            | heterogeneous_foa |
|                 5 | logiqa               |                50 |             0.62   |            0.6    | 62.0%             | 60.0%            | reagents          |
|                 6 | hle                  |                50 |             0.82   |            0.52   | 82.0%             | 52.0%            | reagents          |
|                 7 | mtsamples_procedures |                50 |             0.704  |            0.7102 | 0.0%              | 2.0%             | heterogeneous_foa |
|                 8 | mimic_rrs            |                50 |             0.7489 |            0.7431 | 0.0%              | 0.0%             | reagents          |
|                 9 | pubmed_qa            |                50 |             0.9    |            0.9    | 90.0%             | 90.0%            | tie               |
|                10 | humaneval            |                50 |             1      |            1      | 100.0%            | 100.0%           | tie               |
|                10 | sonnetwriting        |                50 |             1      |            1      | 100.0%            | 100.0%           | tie               |

### Interpretation

- **Hardest by normalized quality:** `matharena`, where both methods obtain zero quality and zero solved instances on the 15 recorded samples.
- **High graded quality but few/no solved medical cases:** `mimic_rrs` and `mtsamples_procedures`. This is not a contradiction: scores around 3.5–3.7 out of 5 normalize to relatively high quality, but the runtime solved thresholds are stringent (`4.0` and `3.8`).
- **Easy for both methods:** `humaneval` and `sonnetwriting`, where both methods reach normalized quality and solved rate of `1.0`.
- **Largest quality advantage for `reagents`:** `hle` (`0.82` versus `0.52`).
- **Benchmarks where `heterogeneous_foa` leads on quality:** `game24`, `hotpotqa`, `scibench`, `mtsamples_procedures`, and narrowly `mimic_rrs`.

---

## Recommended thesis reporting language

A defensible conclusion based on these logs is:

> Across 11 benchmarks and 515 actually recorded evaluations per method, `reagents` obtained the highest observed effectiveness, achieving the best macro normalized quality (0.6048) and the most solved instances (260/515). `heterogeneous_foa` followed closely in effectiveness (0.5921 macro normalized quality; 254/515 solved) while reducing total monetary cost by 27.79% and total model calls by 26.34% relative to `reagents`. A paired benchmark-stratified bootstrap confidence interval for their macro-quality difference includes zero, indicating that the observed effectiveness difference is not clearly separated in the available logs. Thus, `reagents` is the observed quality leader, whereas `heterogeneous_foa` is the strongest high-quality efficiency trade-off.

Avoid claiming that these metrics prove a preferred method is universally superior: only one logged run per method–benchmark pair is present, the benchmark suite is heterogeneous, and the composite rank depends on declared weights.

---

## Methodological sources and how they support this evaluation

| Evaluation choice | Supporting source | Relevance |
|---|---|---|
| Reporting effectiveness and efficiency together rather than only accuracy | Liang et al., *Holistic Evaluation of Language Models (HELM)*, 2022. [arXiv:2211.09110](https://arxiv.org/abs/2211.09110) | HELM explicitly argues for multi-metric language-model evaluation, including accuracy-like and efficiency dimensions. |
| Cost-aware interpretation of LLM methods | Chen, Zaharia, and Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, 2023/2024. [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) | Motivates examining quality–cost trade-offs rather than quality alone for paid LLM inference. |
| Comparing methods over multiple benchmarks using benchmark-level ranks | Demšar, *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR 7, 2006. [JMLR PDF](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf) | Supports dataset-level comparison and average-rank-style analysis when multiple methods are evaluated over multiple data sets. |
| Bootstrap uncertainty intervals | Efron, *Bootstrap Methods: Another Look at the Jackknife*, The Annals of Statistics 7(1), 1979. [DOI: 10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) | Establishes bootstrap resampling for estimating uncertainty of statistics without strong parametric assumptions. |
| Significance-testing caution in NLP-style evaluations | Dror, Baumer, Shlomov, and Reichart, *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*, ACL 2018. [ACL Anthology](https://aclanthology.org/P18-1128/) | Motivates transparent uncertainty reporting and care in selecting statistical procedures for NLP evaluation metrics. |
| Pareto reasoning for multiple competing objectives | Deb, Pratap, Agarwal, and Meyarivan, *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*, IEEE Transactions on Evolutionary Computation 6(2), 2002. [DOI: 10.1109/4235.996017](https://doi.org/10.1109/4235.996017) | Provides the standard multi-objective notion of non-dominated/Pareto-optimal solutions used to interpret quality–resource trade-offs. |
| Harmonic-mean aggregation principle | van Rijsbergen, *Information Retrieval*, 2nd ed., 1979; see also Christen et al., *A Review of the F-Measure*, ACM Computing Surveys, 2023. [DOI: 10.1145/3606367](https://doi.org/10.1145/3606367) | Supports using a harmonic mean when a combined measure should penalize imbalance between two desirable quantities; here this principle is adapted to quality and solve rate. |

---

## Generated supporting artifacts

The accompanying analysis script outputs the following files:

- `method_summary_metrics.csv`: method-level aggregate metrics.
- `metric_rankings.csv`: ranking of every method under each metric.
- `benchmark_method_metrics.csv`: per-benchmark results for all methods.
- `focus_methods_benchmark_difficulty.csv`: benchmark difficulty summary for `reagents` and `heterogeneous_foa`.
- `top_quality_bootstrap.csv`: top-two quality-difference bootstrap interval.
- `plots/overall/`: aggregate method-comparison figures.
- `plots/per_benchmark/`: one focus-method comparison figure for each benchmark.
- `plots/dashboards/`: aggregated benchmark subplot figures.

