# CSV Orientation Experiment

Measures whether LLMs answer questions about tabular data more accurately when the CSV is presented in **row-wise** (standard) vs **column-wise** (transposed) format.

Each evaluation task is presented to the model twice — once with each orientation — and scored independently, so the delta is directly attributable to layout rather than question difficulty.

## How it works

For each run, the framework:

1. Samples `n` tasks from the dataset (seeded, reproducible)
2. For each task, builds two prompts — one row-wise, one column-wise
3. Sends both prompts to the model and scores each response
4. Writes results to `results/` as JSON

### Task types

| Type | Question | Scoring |
|---|---|---|
| `cell_recall` | What is the `{col}` of `{entity}`? | Exact match (normalised) |
| `attr_scan` | Which `{id_col}` has `{col}` equal to `{value}`? | Exact match |
| `comparison` | Which `{id_col}` has the highest `{col}`? | Exact match |
| `row_list` | List all attributes of `{entity}` as key=value pairs. | Subset match (order-independent) |

Tasks cycle round-robin through all four types. Each task receives its own random slice of up to 15 rows as context.

### CSV orientations

**Row-wise** (standard):
```
content_id,title,type
C100271,Neon Streets,Movie
C101244,Dark Signal,Series
```

**Column-wise** (transposed):
```
content_id,C100271,C101244
title,Neon Streets,Dark Signal
type,Movie,Series
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For Anthropic models, add your API key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## Running an experiment

```bash
python -m src.run \
  --model qwen2.5:3b \
  --id-col content_id \
  --n 50 \
  data/raw/ott_movies_subset.csv
```

Results are written to `results/{dataset}_{timestamp}.json`.

### Supported models

| Model | Provider | Notes |
|---|---|---|
| `qwen2.5:3b` | Ollama | Requires local Ollama service |
| `qwen3:8b` | Ollama | Requires local Ollama service |
| `llama3:8b` | Ollama | Requires local Ollama service |
| `claude-haiku-4-5` | Anthropic | Uses Batch API |
| `claude-sonnet-4-6` | Anthropic | Uses Batch API |
| `claude-opus-4-8` | Anthropic | Uses Batch API |

### Adding a model

```bash
python -m src.registry add qwen3:14b ollama
python -m src.registry add claude-haiku-4-5-20251001 anthropic
python -m src.registry list
python -m src.registry remove qwen3:14b
```

Models are stored in `models.json` at the project root. The runner is selected automatically from the registry — no code changes needed.

## Viewing results

```bash
python -m src.report results/*.json
```

Example output:

```
dataset        model          task_type       row_acc  col_acc    delta    n
------------------------------------------------------------------------------------
ott_movies_subset qwen2.5:3b  attr_scan        20.0%    20.0%    +0.0%   50
ott_movies_subset qwen2.5:3b  cell_recall      84.0%    72.0%   -12.0%   50
ott_movies_subset qwen2.5:3b  comparison       56.0%    48.0%    -8.0%   50
ott_movies_subset qwen2.5:3b  row_list         64.0%    48.0%   -16.0%   50
------------------------------------------------------------------------------------
ott_movies_subset (all)        TOTAL            56.0%    47.0%    -9.0%  200
```

`delta = col_acc - row_acc`. A negative delta means row-wise orientation performed better for that task type.

## Running tests

```bash
pytest
```

The test suite does not call any LLM. It tests prompt construction, task generation, scoring logic, and runner mechanics using hardcoded fixtures and mocks.

## Project structure

```
models.json          Model → provider registry
data/raw/            Input CSV datasets
results/             Output JSON from experiment runs
src/
  run.py             Unified CLI entry point
  registry.py        CLI to manage models.json
  runners/
    base.py          BaseRunner ABC
    ollama_runner.py Synchronous local inference via Ollama
    anthropic_runner.py Batch API inference via Anthropic
    factory.py       RunnerFactory — routes by model name
  tasks.py           Task generation (4 types)
  prompt.py          Prompt construction (row/col orientations)
  score.py           Answer scoring
  orient.py          CSV formatting utilities
  report.py          Result aggregation and display
tests/               Unit tests (no LLM calls)
```
# OrientBench
