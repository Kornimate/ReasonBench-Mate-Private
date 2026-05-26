# V2 Metrics Summary
## Important terming

As we are progressing further 2 new terminology is introduced for methods:

 - het_foa $\rightarrow$ heterogeneous_foa
 - new_algo $\rightarrow$ reagents

## Metric Definitions

Let:

- $m$ denote a method. (like rap, foa, ect.)
- $d \in D$ denote a benchmark, with $|D| = 11$. (like mtsamples, mimic_rrs, hle, game24 ect.)
- $i$ denote an evaluated sample within a benchmark. (a sample of the 50 run samples per benchmark)
- $q_{m,d,i}$ denote the recorded raw quality score. (in the logs the list of results where the list contains quality scores)
- $M_d$ denote the maximum score scale used for normalization, with $M_d = 5$ for some of the medical tasks and $M_d = 1$ otherwise. (max score for the benchmark: mtsamples, mimic_rrs: 1-5, otherwise 0-1)
- $\tau_d$ denote the solved threshold for benchmark $d$. (each benchmark has a threshold for quality score when it is marked as solved, if bigger than threshold)
- $n_d$ denote the number of evaluated samples in benchmark $d$. (15 for matharena otherwise 50)
- $C_m$ and $A_m$ denote total dollar cost and total model calls for method $m$. (got from logs as well)

### 1. Normalized per-sample quality

$$
\tilde{q}_{m,d,i} = \frac{q_{m,d,i}}{M_d}.
$$

This maps all benchmark quality values to a comparable $[0, 1]$ range.

### 2. Per-benchmark normalized quality

$$
Q_{m,d} = \frac{1}{n_d} \sum_{i=1}^{n_d} \tilde{q}_{m,d,i}.
$$

### 3. Macro normalized quality

$$
Q^{\mathrm{macro}}_m =
\frac{1}{|D|} \sum_{d \in D} Q_{m,d}.
$$

Every benchmark contributes equally, so `matharena` does not become less important simply because it has fewer evaluated samples. This is the primary overall quality measure.

### 4. Micro normalized quality

$$
Q^{\mathrm{micro}}_m =
\frac{\sum_{d \in D} \sum_{i=1}^{n_d} \tilde{q}_{m,d,i}}
{\sum_{d \in D} n_d}.
$$

Every evaluated instance contributes equally; consequently, benchmarks with 50 samples contribute more than `matharena`, which has 15.

### 5. Solved instances and solve rates

$$
S_{m,d} =
\sum_{i=1}^{n_d}
\mathbf{1}\left[q_{m,d,i} \geq \tau_d\right],
\qquad
SR_{m,d} = \frac{S_{m,d}}{n_d}.
$$

Total solved instances and the micro solve rate are:

$$
S_m = \sum_{d \in D} S_{m,d},
\qquad
SR^{\mathrm{micro}}_m =
\frac{S_m}{\sum_{d \in D}n_d}.
$$

Macro solve rate gives each benchmark equal weight:

$$
SR^{\mathrm{macro}}_m =
\frac{1}{|D|} \sum_{d \in D} SR_{m,d}.
$$

### 6. Harmonic effectiveness

To require strength in both graded quality and thresholded solving, the script computes:

$$
H_m =
\frac{2 Q^{\mathrm{macro}}_m SR^{\mathrm{macro}}_m}
{Q^{\mathrm{macro}}_m + SR^{\mathrm{macro}}_m}.
$$

This metric is inspired by harmonic-mean measures such as the F-measure: a method scores highly only when both components are high.

### 7. Resource-efficiency metrics

$$
\mathrm{CostPerSolved}_m = \frac{C_m}{S_m},
\qquad
\mathrm{CallsPerSolved}_m = \frac{A_m}{S_m},
$$

$$
\mathrm{QualityPerDollar}_m =
\frac{Q^{\mathrm{macro}}_m}{C_m},
\qquad
\mathrm{QualityPer1000Calls}_m =
\frac{Q^{\mathrm{macro}}_m}{A_m / 1000},
$$

$$
\mathrm{SolvedPerDollar}_m =
\frac{S_m}{C_m},
\qquad
\mathrm{SolvedPer1000Calls}_m =
\frac{S_m}{A_m / 1000}.
$$

These resource-normalized metrics favor cheap single-pass methods such as `io`, these metrics may not be interpreted alone due to the different ways (some methods spawn way more states than others) methods handle states. But rather as benchmark for similar ones like (foa-heterogeneous_foa-reagents-rap-tot_bfs)

### 8. Performance-priority composite rank

The script ranks each method on macro quality, total solved instances, total cost, and total calls, and combines the ranks as:

$$
R^{\mathrm{composite}}_m =
0.40R^Q_m +
0.30R^S_m +
0.15R^C_m +
0.15R^A_m,
$$

where quality and solved ranks are descending-performance ranks, while cost and call ranks prefer smaller values. The weighting gives **70%** of the aggregate importance to effectiveness and **30%** to resource use.

| Rank | Method | Composite-rank value |
|---:|---|---:|
| 1 | `reagents` | 2.80 |
| 2 | `heterogeneous_foa` | 2.90 |

This metric is reported with its explicit weights, as it is a declared decision rule 

### 9. Four-dimensional Pareto frontier

Method $a$ dominates method $b$ when it is at least as good on both effectiveness dimensions and no more expensive on both resource dimensions, with a strict improvement in at least one dimension:

$$
Q^{\mathrm{macro}}_a \ge Q^{\mathrm{macro}}_b,
\quad
S_a \ge S_b,
\quad
C_a \le C_b,
\quad
A_a \le A_b.
$$

The Pareto-optimal methods in these logs are:

| Method | Role in the frontier |
|---|---|
| `reagents` | Highest observed effectiveness |
| `heterogeneous_foa` | High-effectiveness, lower-resource trade-off |
| `io` | Very low-resource, lower-quality trade-off |

### 10. Paired benchmark-stratified bootstrap uncertainty

For the two methods (heterogeneous_foa, reagents), a paired difference is computed for every shared sample in each benchmark:

$$
\Delta_{d,i} =
\tilde{q}_{\text{reagents},d,i}
-
\tilde{q}_{\text{heterogeneous\_foa},d,i}.
$$

Within each benchmark, paired differences are resampled with replacement. The mean difference is computed per benchmark and then averaged across benchmarks, maintaining macro benchmark weighting.
