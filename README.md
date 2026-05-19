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

# All options
python experiments/run.py --help
```

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

## Running tests

```bash
pytest
```