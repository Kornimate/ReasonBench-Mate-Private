# Metrics Summary

This document summarizes the metrics implemented in:

- `src/visualization/log_metrics.py`
- `src/visualization/dataset_metrics.py`

It describes what each metric measures, what data it is calculated from, and the formulas used by the implementation.

## Notation

- `N`: number of rows, samples, responses, or events in the relevant group.
- `c_i`: count of item `i`.
- `p_i = c_i / sum_j c_j`: empirical probability of item `i`.
- `H(X) = -sum_i p_i log2(p_i)`: Shannon entropy.
- `H_norm(X) = H(X) / log2(K)`, where `K` is the number of labels/items with non-zero count. If `K <= 1`, the implementation returns `0.0`.
- `mean(x) = sum_i x_i / N`.
- `median(x)`: median value of the sample.
- Ratios with missing values or zero denominators are reported as missing/null.

## Data Sources

### Log Metrics

`log_metrics.py` reads experiment logs from these subdirectories under the configured log path:

- `logs/simple`
- `logs/repeats`
- `logs/cached`

Each method log is parsed into a `MethodLog` with inferred context:

- `model`
- `benchmark`
- `method`
- `split`
- `repeat`

The script reads:

- Per-sample quality lines matching `Sample <n>: solved=<bool> score=<score>`.
- Aggregate quality lines matching `Average correctness: <score>`.
- Structured method events matching `<EVENT_LABEL> {json payload}`.
- Usage accounting lines for `Calls`, `Tokens`, and `Cost`.

Raw-call consistency metrics additionally read raw LLM call logs from the configured raw path, usually `logs/raw_calls`.

### Dataset Metrics

`dataset_metrics.py` loads benchmark datasets through each task's benchmark class. It currently defines dataset specs for:

- `game24`
- `hle`
- `hotpotqa`
- `humaneval`
- `scibench`
- `sonnetwriting`
- `logiqa`
- `matharena`
- `pubmed_qa`
- `mimic_rrs`
- `mtsamples_procedures`

For every loaded sample, the script extracts:

- `input_text`
- `answer_text`
- tokenized input words
- tokenized answer words
- character counts
- normalized answer label

Tokens are extracted with the regex `[A-Za-z0-9_]+` after lowercasing.

## Log Metrics

### Final Score

Output column:

- `final_score`

Description:

The main quality score for a method run.

Data used:

- Prefer `Average correctness` from the log.
- If absent, use the mean of parsed per-sample `score` values.
- If both are absent, use the maximum terminal score found in structured event payloads.

Formula:

```text
final_score =
  avg_score, if present
  mean(sample_scores), if per-sample scores are present
  max(terminal_scores), otherwise if terminal event scores are present
```

### Effort To Solution

Output file:

- `effort_to_solution.csv`

Output columns:

- `event_count`
- `method_effort`
- `steps_observed`
- `steps_to_first_solution`
- `calls_total`
- `score_per_method_effort`
- `score_per_call`

Description:

These metrics measure how much logged method work was needed for a run and how efficiently that work translated into score.

Data used:

- Structured method events from log lines such as `REAGENTS_STEP {...}`.
- Event step fields: `step`, `iteration`, or `depth`.
- Event effort fields: `proposal_count`, `selected_count`, `frontier_size`, `width`, `num_agents`, `root_visits`, `root_children`.
- Summary count fields under `actions`, `generated`, and `aggregated`.
- Parsed `Calls (total)` line.

Formulas:

```text
event_count = number of structured events

event_effort(e) =
  sum(max(0, e[k])) for k in {
    proposal_count, selected_count, frontier_size, width,
    num_agents, root_visits, root_children
  }
  + sum(e[s]["count"]) for s in {actions, generated, aggregated}

method_effort = sum_e event_effort(e)

steps_observed = max(parsed_step) + 1

steps_to_first_solution = min(step where event.solved is True) + 1

score_per_method_effort = final_score / method_effort

score_per_call = final_score / calls_total
```

### Usage, Token, Cost, And Call Metrics

Output file:

- `usage_metrics.csv`

Output columns:

- `calls_total`
- `calls_saved_by_cacher`
- `calls_saved_by_deduplicator`
- `calls_without_cache_or_dedup`
- `input_tokens`
- `output_tokens`
- `cached_tokens`
- `total_tokens`
- `input_tokens_saved_by_cacher`
- `output_tokens_saved_by_cacher`
- `tokens_saved_by_cacher`
- `input_tokens_saved_by_deduplicator`
- `output_tokens_saved_by_deduplicator`
- `tokens_saved_by_deduplicator`
- `total_tokens_with_saved`
- `token_savings_ratio`
- `input_cost`
- `output_cost`
- `total_cost`
- `input_cost_saved_by_cacher`
- `output_cost_saved_by_cacher`
- `total_cost_saved_by_cacher`
- `input_cost_saved_by_deduplicator`
- `output_cost_saved_by_deduplicator`
- `total_cost_saved_by_deduplicator`
- `total_cost_with_saved`
- `cost_savings_ratio`
- `score_per_call`
- `score_per_1k_tokens`
- `score_per_dollar`
- `cost_per_call`
- `tokens_per_call`

Description:

These metrics summarize API usage, cache/deduplication savings, and cost efficiency.

Data used:

- `Calls (total): <int>`
- `Calls (saved by cacher): <int>`
- `Calls (saved by deduplicator): <int>`
- `Tokens (total): {'in': ..., 'out': ..., 'cached': ...}`
- `Tokens (saved by cacher): {'in': ..., 'out': ...}`
- `Tokens (saved by deduplicator): {'in': ..., 'out': ...}`
- `Cost (total): {'in': ..., 'out': ..., 'total': ...}`
- `Cost (saved by cacher): {'in': ..., 'out': ..., 'total': ...}`
- `Cost (saved by deduplicator): {'in': ..., 'out': ..., 'total': ...}`
- `final_score`

Formulas:

```text
calls_without_cache_or_dedup =
  calls_total + calls_saved_by_cacher + calls_saved_by_deduplicator

total_tokens = input_tokens + output_tokens

tokens_saved_by_cacher =
  input_tokens_saved_by_cacher + output_tokens_saved_by_cacher

tokens_saved_by_deduplicator =
  input_tokens_saved_by_deduplicator + output_tokens_saved_by_deduplicator

total_tokens_with_saved =
  total_tokens + tokens_saved_by_cacher + tokens_saved_by_deduplicator

token_savings_ratio =
  (tokens_saved_by_cacher + tokens_saved_by_deduplicator)
  / total_tokens_with_saved

total_cost_with_saved =
  total_cost + total_cost_saved_by_cacher + total_cost_saved_by_deduplicator

cost_savings_ratio =
  (total_cost_saved_by_cacher + total_cost_saved_by_deduplicator)
  / total_cost_with_saved

score_per_call = final_score / calls_total

score_per_1k_tokens = final_score / (total_tokens / 1000)

score_per_dollar = final_score / total_cost

cost_per_call = total_cost / calls_total

tokens_per_call = total_tokens / calls_total
```

### Exploration Diversity

Output file:

- `exploration_diversity.csv`

Output columns:

- `action_entropy`
- `normalized_action_entropy`
- `mean_unique_action_ratio`
- `observed_action_mass`

Description:

These metrics estimate how diverse the method's explored/generated actions were.

Data used:

- Structured event summaries under `actions`, `generated`, and `aggregated`.
- For each summary, the script uses `count`, `unique`, and `most_common`.

Formulas:

```text
unique_action_ratio(summary) = summary["unique"] / summary["count"]

mean_unique_action_ratio =
  mean(unique_action_ratio(summary)) over summaries with count > 0

action_count[a] =
  sum of counts for action a across all summary["most_common"] entries

action_entropy = -sum_a p_a log2(p_a)

normalized_action_entropy =
  action_entropy / log2(number of actions with non-zero count)

observed_action_mass = sum_a action_count[a]
```

### Convergence AUC

Output file:

- `convergence_auc.csv`

Output columns:

- `convergence_auc`
- `final_logged_best`
- `solved_step`
- `trajectory`

Description:

These metrics describe how quickly the logged best score improves over method steps.

Data used:

- Step fields: `step`, `iteration`, or `depth`.
- Numeric summary values under `values`, `new_values`, `old_values`.
- Terminal scores under `terminal.scores`.
- Event field `solved`.

Formulas:

```text
scores_at_step[s] =
  all numeric values logged for step s

running_best_s =
  max(running_best_{s-1}, max(scores_at_step[s]))

trajectory = [(s, running_best_s) for each logged step s]

convergence_auc =
  sum_s running_best_s / max(1, last_logged_step + 1)

final_logged_best = last value in the running-best trajectory

solved_step = first step where event.solved is True
```

### Raw Response Consistency

Output file:

- `raw_response_consistency.csv`

Output columns:

- `prompt_key`
- `prompt_preview`
- `response_count`
- `unique_answer_count`
- `answer_entropy`
- `normalized_answer_entropy`
- `majority_agreement`
- `mean_pairwise_edit_distance`
- `mean_response_length`

Description:

These metrics measure stability across repeated responses to the same raw prompt.

Data used:

- Raw-call logs split by the marker:

```text
********
* USER *
********
```

- The prompt text before `* N: <n> *`.
- Response blocks matching `* RESPONSE <n> *`.
- Context inferred from the raw-call log path.

Answer normalization:

- If a response contains standalone `A`, `B`, or `C`, that letter is used.
- Otherwise, whitespace is normalized, text is lowercased, and the first 160 characters are used.

Formulas:

```text
prompt_key = first 16 hex chars of SHA256(prompt)

response_count = number of responses grouped under a prompt_key

unique_answer_count = number of unique normalized answers

answer_entropy = -sum_a p_a log2(p_a)

normalized_answer_entropy =
  answer_entropy / log2(number of answers with non-zero count)

majority_agreement =
  max_a count(a) / response_count

normalized_edit_distance(x, y) =
  levenshtein(x, y) / max(len(x), len(y), 1)

mean_pairwise_edit_distance =
  mean(normalized_edit_distance(x_i, x_j)) over all response pairs i < j

mean_response_length = mean(len(response))
```

## Dataset Metrics

### Per-Sample Dataset Fields

Output file:

- `dataset_samples.csv`

Output columns:

- `dataset`
- `sample_idx`
- `input_text`
- `answer_text`
- `input_word_count`
- `answer_word_count`
- `input_char_count`
- `answer_char_count`
- `label`

Description:

These are per-instance measurements used to build the aggregate dataset metrics.

Data used:

- Dataset rows loaded through task benchmark classes.
- Dataset-specific extractors in `dataset_metrics.py`.

Formulas:

```text
input_word_count = number of regex tokens in input_text

answer_word_count = number of regex tokens in answer_text

input_char_count = len(input_text)

answer_char_count = len(answer_text)
```

Answer labels are normalized as:

```text
label =
  "<empty>" if answer is empty
  "A", "B", "C", or "D" if answer exactly matches those labels
  "yes", "no", or "maybe" if answer matches those words case-insensitively
  lowercase answer if len(answer) <= 32
  "<free_text>" otherwise
```

### Aggregate Dataset Metrics

Output file:

- `dataset_metrics.csv`

Output columns:

- `status`
- `num_instances`
- `median_input_words`
- `mean_input_words`
- `median_answer_words`
- `mean_answer_words`
- `input_lexical_diversity`
- `answer_entropy`
- `normalized_answer_entropy`
- `dominant_label_share`
- `free_text_answer_share`

Description:

These metrics summarize dataset size, input/answer length, lexical diversity, and answer-space shape.

Data used:

- Per-sample rows from `dataset_samples.csv`.
- Dataset load failures, when a dataset cannot be imported or loaded.

Formulas:

```text
status =
  "loaded" if the dataset loaded successfully
  error message otherwise

num_instances = number of loaded samples

median_input_words = median(input_word_count)

mean_input_words = mean(input_word_count)

median_answer_words = median(answer_word_count)

mean_answer_words = mean(answer_word_count)

input_lexical_diversity =
  number of unique input tokens across the dataset
  / total number of input tokens across the dataset

answer_entropy = -sum_l p_l log2(p_l)

normalized_answer_entropy =
  answer_entropy / log2(number of labels with non-zero count)

dominant_label_share =
  max_l count(label = l) / num_instances

free_text_answer_share =
  count(label = "<free_text>") / num_instances
```

### Dataset Complexity Heatmap

Output plot:

- `complexity_heatmap.png`

Description:

This visualization combines several aggregate dataset metrics into a min-max normalized matrix. It is a plot-only derived view, not a separate CSV.

Data used:

- `median_input_words`
- `median_answer_words`
- `input_lexical_diversity`
- `normalized_answer_entropy`
- `dominant_label_share`

Formula:

```text
normalized_value =
  (value - column_min) / (column_max - column_min)
```

If `column_max - column_min` is zero, the denominator is replaced with `1`.

## Plot Outputs

`log_metrics.py` writes:

- `effort_to_solution.png`: bar plot of mean `score_per_method_effort` grouped by benchmark and method.
- `exploration_diversity.png`: bar plot of mean `normalized_action_entropy` grouped by benchmark and method.
- `convergence_auc.png`: mean running-best score trajectory by benchmark and method.
- `calls_total.png`: bar plot of mean `calls_total`.
- `total_tokens.png`: bar plot of mean `total_tokens`.
- `total_cost.png`: bar plot of mean `total_cost`.
- `score_per_dollar.png`: bar plot of mean `score_per_dollar`.
- `raw_response_consistency.png`: scatter plot of `majority_agreement` vs `mean_pairwise_edit_distance`, colored by `normalized_answer_entropy`.
- `raw_majority_agreement.png`: bar plot of mean `majority_agreement`.

`dataset_metrics.py` writes:

- `dataset_size.png`: bar plot of `num_instances`.
- `input_length_distribution.png`: boxplot of per-sample `input_word_count`.
- `answer_length_vs_entropy.png`: scatter plot of `median_answer_words` vs `normalized_answer_entropy`.
- `lexical_diversity.png`: line plot of `input_lexical_diversity`.
- `complexity_heatmap.png`: min-max normalized dataset profile heatmap.

## Reference Sources

These references motivate the metric families used in the scripts:

- Shannon entropy for diversity and answer-space uncertainty: https://en.wikipedia.org/wiki/Entropy_(information_theory)
- Type-token ratio and lexical diversity: https://en.wikipedia.org/wiki/Lexical_diversity
- Type-token ratio overview: https://www.sketchengine.eu/glossary/type-token-ratio-ttr/
- Lexical diversity caveats and alternatives: https://pmc.ncbi.nlm.nih.gov/articles/PMC6037356/
- Anytime algorithms and quality-over-budget evaluation: https://en.wikipedia.org/wiki/Anytime_algorithm
- Performance profiles for algorithm comparison: https://en.wikipedia.org/wiki/Performance_profile
- Area-under-curve framing: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
- Levenshtein/edit distance: https://en.wikipedia.org/wiki/Levenshtein_distance
- Self-consistency for LLM reasoning samples: https://arxiv.org/abs/2203.11171
- Google Research page for self-consistency: https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/
- Sample size and statistical power: https://en.wikipedia.org/wiki/Sample_size_determination
- Statistical power: https://en.wikipedia.org/wiki/Statistical_power
- Long-context benchmark framing: https://arxiv.org/abs/2210.15424
- LongBench: https://arxiv.org/abs/2307.11088
- Language model benchmark overview: https://en.wikipedia.org/wiki/Language_model_benchmark
- Natural Questions: https://ai.google.com/research/NaturalQuestions
- Answer-type framing in QA tasks: https://arxiv.org/abs/1906.00300
- Class imbalance: https://en.wikipedia.org/wiki/Class_imbalance_problem
- Dataset bias in reading comprehension: https://arxiv.org/abs/1906.07413
