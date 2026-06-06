# ReAgents V2 Variants

## Source Paper Basis

These implementations are based on the repository's existing `src/methods/reagents.py` implementation. That method follows the same broad paper lineage as:

- **ReAct: Synergizing Reasoning and Acting in Language Models** by Yao et al.  
  https://arxiv.org/abs/2210.03629
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** by Yao et al.  
  https://arxiv.org/abs/2305.10601

The base ReAgents method combines repeated state expansion, LLM-generated actions, evaluator scoring, adaptive priors over step-agent types, width adaptation, backtracking, and resampling. The V2 variants keep that search loop unchanged and modify only the agent-selection mechanism.

## Implemented Variants

### `reagents_V2comp`

Comparative selection converts the current depth-specific ReAgents priors into pairwise comparisons between step-agent types. Each candidate agent is scored by how often it would win against the other candidates under the current prior distribution, then sampled from the resulting comparative probabilities.

Useful when you want a softer relative-preference selector that still explores while favoring agents that are consistently better than their alternatives.

### `reagents_v2tour`

Tournament selection samples a small set of candidate step-agent types from the current ReAgents priors and selects the strongest candidate inside that sampled tournament. A small epsilon exploration rate allows occasional random selection.

Useful when you want a more decisive selector that amplifies the best current prior without fully collapsing exploration.

## Configuration

Both variants inherit the normal `reagents` config if no method-specific YAML block exists. Optional extra parameters:

- `comparison_temperature`, default `0.25`
- `comparison_epsilon`, default `0.05`
- `tournament_size`, default `2`
- `tournament_epsilon`, default `0.05`
