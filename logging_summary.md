# Method Logging Summary

Method-level logs are written to the main experiment log handler. Each instrumented section starts with a title line, followed by tab-prefixed JSON rows, then an empty row.

Raw call logging is unchanged; these logs describe method-internal search, voting, selection, and adaptation metrics.

## IO

Title: `IO Method Information:`

Rows:
- `IO_RESULT`: `idx`, `n`, action count/unique/most-common summary, final state depths, terminal count, solved count, score summary.

## CoT

Title: `CoT Method Information:`

Rows:
- `COT_RESULT`: `idx`, `n`, action count/unique/most-common summary, final state depths, terminal count, solved count, score summary.

## CoT-SC

Title: `CoT-SC Method Information:`

Rows:
- `COT_SC_RESULT`: `idx`, sample count `n`, action summary, full vote counts, selected vote count, terminal count, solved count, score summary.

## ReAct

Title: `ReAct Method Information:`

Rows:
- `REACT_STEP`: `idx`, `step`, action summary, state depth, final flag, score, solved flag. If a step errors, logs the action summary and error string.
- `REACT_RESULT`: final terminal count, solved count, score summary.

## FoA

Title: `FoA Method Information:`

Rows:
- `FOA_STEP`: `idx`, `step`, number of agents, action summary, failed/final state count, replacement count, visited state count, evaluation score summary when evaluation ran, resampled count, state depths, terminal count, solved count, score summary, solved flag.

## Heterogeneous FoA

Title: `Heterogeneous FoA Method Information:`

Rows:
- `HETEROGENEOUS_FOA_STEP`: `idx`, `step`, number of agents, configured agent types, action summary, finished/final state count, replacement count, visited state count, evaluation score summary when evaluation ran, resampled count, state depths, terminal count, solved count, score summary, solved flag.

## ToT-BFS

Title: `ToT-BFS Method Information:`

Rows:
- `TOT_BFS_STEP`: `idx`, `step`, frontier size, proposal count, selected count, action summary, value score summary, terminal count, solved count, score summary, solved flag. Empty proposal steps include `empty_proposals`.

## ToT-DFS

Title: `ToT-DFS Method Information:`

Rows:
- `TOT_DFS_STEP`: `idx`, depth, frontier size, proposal count, selected count, pruned count, action summary, value score summary, pruning threshold.
- `TOT_DFS_LEAF`: `idx`, depth, iteration count, frontier size, proposal count, action summary, value score summary, best value, terminal count, solved count, score summary, solved flag.
- `TOT_DFS_STOP`: stop reason and iteration limit information.
- `TOT_DFS_ERROR`: error string if the DFS wrapper catches an exception.
- `TOT_DFS_RESULT`: final iteration count and output count.

## GoT

Title: `GoT Method Information:`

Rows:
- `GOT_STEP`: `idx`, `step`, frontier size, generated action summary, aggregated action summary, proposal count, selected count, value score summary, terminal count, solved count, score summary, solved flag. Empty proposal steps include `empty_proposals`.

## RAP

Title: `RAP Method Information:`

Rows:
- `RAP_ITERATION`: `idx`, iteration, selected node depth, selected node visits before update, current node depth, root visits, root child count, node child count, simulated value, best value, best action count, terminal count, solved count, score summary. If solved during expansion, logs expansion-specific terminal/reward data.
- `RAP_RESULT`: `idx`, best action count, final flag, reward, solved flag.

## ReAgents

Title: `ReAgents Method Information:`

Rows:
- `REAGENTS_STEP`: `idx`, `step`, current width, next width, selected agent indices/types/counts, old value summary, new value summary, terminal indices, solved indices, prior matrix, terminal count, solved count, score summary.

## ReAgents AOS

Title: `Runtime Agent Distribution Information:`

Rows:
- `AOS_STEP`: `step`, width, agent types, fleet counts, per-type counts and distribution, AOS Q-values, selection probabilities, average reward by agent type, `avg_reward_act`, `avg_reward_react`, solved flag.

## ReAgents ALNS

Title: `Runtime Agent Distribution Information:`

Rows:
- `ALNS_STEP`: `step`, width, fleet counts, current weights, probabilities, per-type agent counts.
- `ALNS_SEGMENT`: `step`, segment scores and segment counts before weight update.
