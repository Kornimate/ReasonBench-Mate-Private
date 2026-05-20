# Metric Sources

These sources support the log-derived metrics implemented in `src/visualization/log_metrics.py`.

## Existing Result Metric

- Average benchmark score / accuracy is the repository's existing primary quality metric. It is read from the experiment logs as `Average correctness` or from per-sample `score` rows.

## Method-Log Metrics

### 1. Effort-to-Solution / Score per Method Effort

Purpose: compare how much useful score a method obtains relative to its logged search work: proposals, actions, frontier size, selected states, width, and similar method-internal operations.

Source framing:
- Anytime/performance-profile evaluation is commonly used to compare optimization/search methods by quality achieved under a budget.
- Source URL: https://en.wikipedia.org/wiki/Anytime_algorithm
- Source URL: https://en.wikipedia.org/wiki/Performance_profile

### 2. Exploration Diversity

Purpose: compare whether methods explore varied actions/candidates or repeatedly collapse onto the same choices. The implementation uses action unique-ratio and Shannon entropy over logged action summaries.

Source framing:
- Shannon entropy measures uncertainty/diversity of a discrete distribution.
- Source URL: https://en.wikipedia.org/wiki/Entropy_(information_theory)

### 3. Convergence AUC

Purpose: compare how quickly methods improve their logged best score over steps/iterations. The implementation computes a running best-score trajectory and averages it over logged step positions.

Source framing:
- Area-under-curve style summaries are standard for comparing trajectories over budgets; anytime algorithms are evaluated by solution quality as computation proceeds.
- Source URL: https://en.wikipedia.org/wiki/Anytime_algorithm
- Source URL: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

## Raw-Call Metric

### Response Consistency

Purpose: compare stability of LLM responses for repeated identical prompts within each model/benchmark/method context. The implementation groups raw-call logs by prompt, normalizes answer labels when possible, and computes majority agreement, answer entropy, and mean pairwise edit distance.

Source framing:
- Self-consistency in reasoning samples is a meaningful signal for LLM reasoning reliability.
- Source URL: https://arxiv.org/abs/2203.11171
- Source URL: https://research.google/pubs/self-consistency-improves-chain-of-thought-reasoning-in-language-models/
- Levenshtein/edit distance is a standard string difference metric.
- Source URL: https://en.wikipedia.org/wiki/Levenshtein_distance

## Dataset Metrics

These sources support the dataset-level metrics implemented in `src/visualization/dataset_metrics.py`.

### 1. Dataset Size

Purpose: compare how many instances each benchmark contributes. This matters for statistical confidence, variance interpretation, and whether benchmark-level averages are based on comparable sample sizes.

Source framing:
- Sample size directly affects estimate precision and statistical power.
- Source URL: https://en.wikipedia.org/wiki/Sample_size_determination
- Source URL: https://en.wikipedia.org/wiki/Statistical_power

### 2. Input Length Distribution

Purpose: compare prompt/context length across datasets. This connects directly to model context burden, token cost, latency, and long-context reasoning difficulty.

Source framing:
- Long-document and long-context NLP benchmarks explicitly use input/document length as a task difficulty and capability dimension.
- Source URL: https://arxiv.org/abs/2210.15424
- Source URL: https://arxiv.org/abs/2307.11088
- Source URL: https://en.wikipedia.org/wiki/Language_model_benchmark

### 3. Answer Length

Purpose: distinguish classification-style tasks from open-ended generation tasks. Longer targets usually imply different evaluation difficulty, more output-token cost, and more room for stylistic variance.

Source framing:
- QA/NLP dataset documentation commonly separates short-answer and long-answer regimes because answer length changes task behavior and evaluation.
- Source URL: https://ai.google.com/research/NaturalQuestions
- Source URL: https://arxiv.org/abs/1906.00300

### 4. Input Lexical Diversity

Purpose: estimate how varied the input language is within each dataset. Higher diversity can explain greater exploration needs, parser brittleness, and prompt-response variability.

Source framing:
- Type-token ratio is a standard lexical diversity measure, though it should be interpreted alongside text length.
- Source URL: https://en.wikipedia.org/wiki/Lexical_diversity
- Source URL: https://www.sketchengine.eu/glossary/type-token-ratio-ttr/
- Source URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6037356/

### 5. Answer-Space Entropy / Dominant Label Share

Purpose: quantify whether answers are concentrated in a few classes or spread across many labels/free-text targets. This helps interpret majority baselines, self-consistency, and response collapse.

Source framing:
- Shannon entropy measures uncertainty/diversity in a discrete distribution.
- Class/label imbalance is a known issue for machine-learning evaluation and training.
- Source URL: https://en.wikipedia.org/wiki/Entropy_(information_theory)
- Source URL: https://en.wikipedia.org/wiki/Class_imbalance_problem
- Source URL: https://arxiv.org/abs/1906.07413
