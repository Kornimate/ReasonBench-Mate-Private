# ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning

**ReasonBENCH** is a benchmark suite and open-source library for controlled multi-run evaluation of LLM reasoning. It measures both the quality and stability of reasoning strategies by running repeated independent trials and reporting variance-aware metrics, including confidence intervals, run deviation, and global noise, rather than relying on single-run averages.

> *Preliminary work. Under review by the International Conference on Machine Learning (ICML).*

<!-- Leaderboard: http://reasonbench.github.io -->

## Motivation

LLM reasoning is typically evaluated using single runs, masking how much performance can vary across repeated executions. This practice obscures both reliability and cost, and can lead to misleading comparisons between methods and models. ReasonBENCH addresses this by repeating every model-strategy-task configuration with 10 independent trials and reporting distributional metrics alongside averages.

Key findings from our evaluation:
- **Run-to-run variability is substantial** - often large enough to change model/method rankings relative to single-run averages
- **Quality and cost stability decouple** - the most accurate strategy is not necessarily the most stable, and vice versa
- **Model scaling improves both quality and stability** - larger models within a family yield tighter distributions
- **Prompt refinements improve quality but not stability** - clarifying prompts and parsers boosts accuracy without reducing run-to-run variance
- **Reasoning effort scales cost, not quality** - increasing test-time reasoning effort primarily raises cost with limited and statistically insignificant quality gains

## Reasoning Strategies

ReasonBENCH implements representative reasoning strategies behind a standardized method interface:

| Strategy | Type | Reference |
|----------|------|-----------|
| **IO** | Direct | - |
| **CoT** | Direct | Wei et al., 2022 |
| **CoT-SC** | Direct | Wang et al., 2023 |
| **ReAct** | Adaptive | Yao et al., 2023b |
| **ToT-BFS** | Structured | Yao et al., 2023a |
| **ToT-DFS** | Structured | Yao et al., 2023a |
| **GoT** | Structured | Besta et al., 2024 |
| **RAP** | Planning | Hao et al., 2023 |
| **FoA** | Evolutionary | Klein et al., 2025 |
| **Het-FoA** | Evolutionary | `shoan-main` heterogeneous fleet variant |
| **ReAgEnTS** | Evolutionary | Adaptive multi-agent reasoning variant |
| **ReAgEnTS-V2 Comp** | Evolutionary | Competition-style ReAgEnTS variant |
| **ReAgEnTS-V2 Tour** | Evolutionary | Tournament-style ReAgEnTS variant |

## Benchmarks

ReasonBENCH includes core reasoning benchmarks plus newer medical, logic, and math tasks:

| Task | CLI name | Domain | Metric |
|------|----------|--------|--------|
| **Game of 24** | `game24` | Mathematical reasoning | Accuracy |
| **SciBench** | `scibench` | Scientific reasoning | Accuracy / exact match |
| **HumanEval** | `humaneval` | Code generation | pass@1 |
| **HotPotQA** | `hotpotqa` | Multi-hop QA | Exact match |
| **Sonnet Writing** | `sonnetwriting` | Creative writing | Rhyme and word constraints |
| **HLE** | `hle` | General reasoning (Humanity's Last Exam) | Accuracy |
| **LogiQA** | `logiqa` | Logical reasoning | Accuracy |
| **MathArena** | `matharena` | Competition math | Accuracy |
| **PubMedQA** | `pubmed_qa` | Biomedical QA | Accuracy |
| **MIMIC-RRS** | `mimic_rrs` | Clinical reasoning | Task-specific score |
| **MTSamples Procedures** | `mtsamples_procedures` | Clinical procedure coding | Task-specific score |

## Evaluated Models

10 contemporary reasoning models from 6 providers:

| Model | Provider |
|-------|----------|
| GPT-4.1 Nano, GPT-4.1 Mini | OpenAI |
| GPT-5 Nano, GPT-5 Mini | OpenAI |
| GPT-OSS 120B | Together AI |
| DeepSeek R1 | Together AI |
| Llama 4 Maverick | Together AI |
| Qwen3-235B Thinking | Together AI |
| Claude Haiku 4.5 | Anthropic |
| Gemini 3 Flash | Google |

## Setup

Create an environment and install the Python dependencies:

```bash
conda create -n crfm-helm python=3.10 pip
conda activate crfm-helm
pip install crfm-helm
pip install -r requirements.txt
```

This repository vendors the `cachesaver` core that was provided in the `shoan-main (1).zip` reference project, so you do not need a separate `pip install cachesaver`. `poethepoet` is included in `requirements.txt` for the helper tasks defined in `pyproject.toml`.

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY_CLAN="sk-..."
# and/or other provider keys
```

Useful Poe tasks after installation:

```bash
poe save
poe save_results
poe log_visualize_v3
poe ds_visualize
poe cl_visualize
```

## Quick Start

The simplest way to run an experiment is via the shell script:

```bash
bash scripts/simple/bash/simple.sh
```

On Windows PowerShell, use:

```powershell
.\scripts\simple\powershell\simple.ps1
```

Edit the variables at the top of the script to change the benchmark, method, model, and split.

For direct invocation:

```bash
python scripts/simple/simple.py \
    --benchmark game24 \
    --method tot_bfs \
    --split mini \
    --provider openai \
    --api_key OPENAI_API_KEY_CLAN \
    --model gpt-4.1-nano \
    --temperature 1.0 \
    --max_completion_tokens 10000 \
    --top_p 1.0 \
    --batch_size 1 \
    --timeout 2.0 \
    --correctness 1 \
    --allow_batch_overflow 1 \
    --ns_ratio 0.0 \
    --value_cache
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--benchmark` | Task name, for example `game24`, `humaneval`, `hotpotqa`, `scibench`, `hle`, `sonnetwriting`, `logiqa`, `matharena`, `pubmed_qa`, `mimic_rrs`, or `mtsamples_procedures` |
| `--method` | Reasoning method: `io`, `cot`, `cot_sc`, `foa`, `heterogeneous_foa`, `reagents`, `reagents_v2comp`, `reagents_v2tour`, `tot_bfs`, `tot_dfs`, `got`, `react`, `rap` |
| `--split` | Dataset split: `full`, `train`, `validation`, `test`, `mini`, `single`, or `n[50]` to take 50 deterministic shuffled instances |
| `--provider` | LLM provider: `openai`, `gemini`, `anthropic`, `groq`, `together` |
| `--model` | Model identifier, such as `gpt-4.1-nano` or `claude-haiku-4-5` |
| `--dataset_path` | Optional explicit dataset file, usually `datasets/dataset_<benchmark>.csv.gz` or the task-specific JSONL path |
| `--ns_ratio` | Namespace ratio from `0.0` to `1.0` for controlling parallel execution |

## Zip Integration Notes

The `shoan-main (1).zip` reference project overlaps with this repository in a very specific way:

- The current repository is the main ReasonBENCH codebase: benchmarks, methods, tasks, models, scripts, and tests all live here.
- The zip contributes the reusable `cachesaver` runtime pieces, plus extra reference material such as ablation outputs and legacy examples.
- The runtime-relevant part has been integrated directly into this repo under `cachesaver/`, and the existing scripts and tests continue to import it as `from cachesaver ...`.

### What was integrated

- `cachesaver/` now contains the async batching, caching, deduplication, reordering, and pipeline code derived from the zip.
- The local `Request` type was adapted to support both the zip's simple request style and this repository's current `args`/`kwargs` request style used by `src/models/api.py`.
- `requirements.txt` now includes `deepdiff`, which is required by both the integrated `cachesaver` hashing logic and the existing API accounting code.

### About `src/methods/cot_sc.py`

`src/methods/cot_sc.py` already belongs to the current repository and does not have a direct counterpart inside the zip. Its role is to:

- sample `n` chain-of-thought completions through the shared model API,
- majority-vote the returned actions,
- apply the selected action once through the environment.

That means the zip does not replace `CoT-SC`; it strengthens the infrastructure underneath it by supplying the local `cachesaver` pipeline that the rest of the repository already expects.

## Evaluation Metrics

For each model-strategy-task configuration, we report metrics along two dimensions:

**Quality:**
- **Average** - stratified bootstrap mean over runs; benchmarks treated as strata
- **Run Deviation** - typical run-to-run deviation from the strategy mean per benchmark
- **Noise (Global)** - variance of z-scored outcomes across all benchmarks
- **Noise (Run)** - average within-benchmark z-score variance

**Cost:**
- Same four metrics computed over token usage and wall-clock time, expressed in USD

## Configuration

Method hyperparameters are defined per task in YAML files under `scripts/configs/`:

```yaml
# scripts/configs/game24.yaml
tot_bfs:
  num_selections: 3
  num_steps: 4
  num_evaluations: 3

got:
  num_selections: 5
  num_steps: 4
  num_generate: 10
  num_evaluations: 3
  num_best: 2
```

Decoding parameters such as temperature, top-p, max tokens, and stop sequences are sourced from `scripts/configs/<task>.env`.

## Architecture

ReasonBENCH is organized around five core abstractions:

- **Method / Framework** - specifies the reasoning strategy independently of the model or task. A method initializes with task agents and an environment, then runs a puzzle instance through a standard solving loop.
- **Agent** - task-agnostic LLM interface used by methods. Agents request model completions, call task prompts, and turn parsed responses into actions, reactions, reflections, or value estimates.
- **Task** - owns task-specific prompts, data loading, state representation, parsing, transition logic, and verification.
- **Model** - uniform interface for LLM providers, supporting async execution and integrated with CacheSaver for response caching and deduplication.
- **CacheSaver** - vendored async request pipeline for batching, caching, deduplication, and deterministic reordering, integrated from the `shoan-main` reference zip.

The task layer is split into:

- **Prompts** - prompt templates for task-specific operations.
- **Data / Benchmark** - dataset loading and split selection.
- **State** - current puzzle state plus action history and task-specific metadata.
- **Environment** - state transitions, heuristic values, response parsing, and final verification.

Most task environments return a verification result with three fields: `finished`, `correct`, and `message`. The message can be used for logging, error analysis, or reflection-style methods.

```
src/
|-- models/          # LLM provider adapters (OpenAI, Anthropic, Groq, Together, Gemini)
|-- methods/         # Reasoning strategy implementations
|-- tasks/           # Task definitions (state, environment, agents, prompts)
|   |-- game24/
|   |-- humaneval/
|   |-- hotpotqa/
|   |-- matharena/
|   |-- mimic_rrs/
|   |-- pubmed_qa/
|   `-- ...
|-- __init__.py      # Factory registrations
|-- typedefs.py      # Core ABCs and type definitions
`-- utils.py         # Logging and utility functions

cachesaver/
|-- batching.py      # Async request batching
|-- caching.py       # Namespace-aware response reuse
|-- deduplicator.py  # Duplicate prompt collapsing
|-- pipelines.py     # Online/local pipeline composition
|-- reordering.py    # Deterministic request ordering
|-- typedefs.py      # Request / Response / protocol types
`-- resource_managers/

scripts/
|-- simple/          # Single-run experiment scripts
|-- repeats/         # Batch/repeated experiment scripts
|-- cached/          # Cached inference scripts
`-- configs/         # YAML and .env configuration files

datasets/            # Gzip-compressed task datasets
tests/               # Pytest test suite
```

## Tests

```bash
pytest                                       # run all tests
pytest tests/got/test_game24.py              # single file
pytest tests/got/test_game24.py -k "test_x"  # single test
```

Tests use async fixtures and require valid API keys (Groq/OpenAI) for the mock LLM clients.

## Citation

```bibtex
@inproceedings{reasonbench2025,
  title={ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning},
  author={Anonymous},
  year={2025},
  note={Under review at ICML}
}
```
