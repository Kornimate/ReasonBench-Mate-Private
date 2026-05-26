# Metric Directions

This file records whether higher or lower values are better when comparing methods.

| Metric | Better Direction | Notes |
| --- | --- | --- |
| `score_per_method_effort` | Higher | More score per unit of method-internal effort. |
| `normalized_action_entropy` | Higher | More diverse explored/generated actions. |
| `convergence_auc` | Higher | Better logged best-score trajectory over steps. |
| `calls_total` | Lower | Fewer model/API calls. |
| `total_tokens` | Lower | Fewer total input/output tokens. |
| `total_cost` | Lower | Lower total API cost. |
| `score_per_dollar` | Higher | More score per USD. |
| `cost_per_solved` | Lower | Lower API cost for each solved sample. |
| `solved_per_dollar` | Higher | More solved samples per USD. |
| `solved_rate_per_dollar` | Higher | More solved-rate per USD spent. |
| `cost_per_solved_rate_point` | Lower | Lower API cost for each solved-rate point. |
| `quality_mean` | Higher | Better average quality score. |
| `solved_rate` | Higher | Larger fraction of solved samples. |
| `mean_solution_time` | Lower | Lower average sample runtime. |
| `mean_solved_solution_time` | Lower | Lower average runtime among solved samples. |
| `clocktime_per_solved` | Lower | Less wall-clock time per solved sample. |
| `majority_agreement` | Higher | More raw-call agreement across repeated responses. |
| `methods_heatmap` | Higher | Cell value is solved status, where `1` is solved. |

Plot calls are commented out in `src/visualization/log_metrics.py` when neither `heterogeneous_foa` nor `reagents` is currently best for that metric.
