# MazeMemory

Master's thesis project (Birkbeck) evaluating whether **shared memory improves LLM-based multi-agent systems**. Three agents navigate an unknown maze independently or with access to a shared memory store, and optionally guided by an observer agent that analyses collective trajectories.

Three scenarios are compared:
- **Baseline** — agents navigate independently with private memory only
- **Shared Memory** — agents write their positions to a shared store and can read others' paths
- **Shared Memory + Observer** — an observer agent synthesises shared trajectories and gives directional recommendations

## Setup

```bash
pip install -e .
cp .env.example .env  # add your API keys
```

## Running experiments

```bash
# Full experiment (180 runs: 3 scenarios × 3 mazes × 20 runs)
python experiments/run.py

# Single test run with live visualization
python experiments/run.py --scenarios baseline --mazes 1 --n-runs 1 --model gpt-4o --provider openai --live --no-db

# Cheapest option (DeepSeek)
python experiments/run.py --scenarios baseline --mazes 1 --n-runs 1 --model deepseek/deepseek-chat --provider deepseek --live --no-db

# All options
python experiments/run.py --help
```

Supported models (pass as `--model`):

| Model | Provider | Input $/1M | Output $/1M |
|---|---|---|---|
| `claude-sonnet-4-6` | `anthropic` | $3.00 | $15.00 |
| `claude-haiku-4-5-20251001` | `anthropic` | $0.80 | $4.00 |
| `gpt-4o` | `openai` | $2.50 | $10.00 |
| `gpt-4o-mini` | `openai` | $0.15 | $0.60 |
| `gpt-5.5` | `openai` | $5.00 | $30.00 |
| `gemini/gemini-2.0-flash` | `gemini` | $0.075 | $0.30 |
| `deepseek/deepseek-chat` | `deepseek` | $0.14 | $0.28 |
| `deepseek/deepseek-v3` | `deepseek` | $0.14 | $0.28 |
| `deepseek/deepseek-v3.2` | `deepseek` | $0.14 | $0.28 |
| `deepseek/deepseek-reasoner` | `deepseek` | $0.435 | $0.87 |

Key flags:

| Flag | Description |
|---|---|
| `--model` | LiteLLM model string (e.g. `gpt-4o`, `claude-sonnet-4-6`) |
| `--provider` | Provider name (`openai`, `anthropic`) |
| `--scenarios` | One or more of `baseline`, `shared_memory`, `shared_memory_observer` |
| `--mazes` | Maze IDs to run (default: all) |
| `--n-runs` | Runs per scenario/maze combination (default: 20) |
| `--live` | Show real-time maze visualization |
| `--no-db` | Skip Supabase, save results locally to `results/` |

## Saving results to Supabase

Runs are always saved locally before any DB write:

```
results/
  experiments/     ← one JSON per run (source of truth for sync)
  shared_memory/   ← one JSONL per run, written incrementally during execution
```

To upload to Supabase:

```bash
# Preview what would be uploaded (no writes)
python experiments/sync.py --dry-run

# Upload all pending runs
python experiments/sync.py
```

Runs already in the database are skipped automatically.

## Running tests

```bash
pytest
```