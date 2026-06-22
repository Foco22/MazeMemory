# MazeMemory

Master's thesis project (Birkbeck) evaluating whether **shared memory improves LLM-based multi-agent systems**. Three agents navigate an unknown maze independently or with access to a shared memory store, and optionally guided by an observer agent that analyses collective trajectories.

Four scenarios are compared in a 2×2 factorial design:

| | No Observer | Observer |
|---|---|---|
| No shared memory (raw) | **Baseline** | **Observer** |
| Shared memory (raw) | **Shared Memory** | **Shared Memory + Observer** |

- **Baseline** (`baseline`) — agents navigate with private memory only; no coordination
- **Shared Memory** (`shared_memory`) — agents write positions to a shared store and can query other agents' full trajectories via `get_shared_memory`
- **Observer** (`observer`) — an observer agent reads the shared store and gives directional recommendations via `get_insight`; agents cannot access the raw store directly
- **Shared Memory + Observer** (`shared_memory_observer`) — agents have both raw shared memory access and observer recommendations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env  # add your API keys
```

> **Note:** always activate the virtual environment before running any command:
> ```bash
> source .venv/bin/activate
> ```
> Without it, Python won't find the `src` package and will throw `ModuleNotFoundError: No module named 'src'`.

## Running experiments

```bash
# Full experiment (180 runs: 3 scenarios × 3 mazes × 20 runs)
python experiments/run.py

# Single test run with live visualization
python experiments/run.py --scenarios baseline --mazes 1 --n-runs 10 --model gpt-4o --provider openai --live --no-db

# DeepSeek V4 Flash (cheap, requires DEEPSEEK_API_KEY in .env)
python -u experiments/run.py --scenarios shared_memory_observer --mazes 1 --n-runs 30 --model deepseek/deepseek-chat --provider deepseek --live --no-db

# Local model via Ollama (free, no API key required)
python -u experiments/run.py --scenarios baseline --mazes 1 --n-runs 1 --model openai/ZimaBlueAI/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF:latest --provider ollama --api-base http://localhost:11434/v1 --api-key ollama --live --no-db

# All options
python experiments/run.py --help
```

### Running 4 models in parallel (one terminal each)

```bash
# Terminal 1 — DeepSeek V4 Pro
python -u experiments/run.py --model deepseek/deepseek-v4-pro --provider deepseek --scenarios baseline --mazes 1 --n-runs 20

# Terminal 2 — Gemini 3.1 Pro Preview
python -u experiments/run.py --model gemini/gemini-3.1-pro-preview --provider google --scenarios baseline --mazes 1 --n-runs 20

# Terminal 3 — Claude Opus 4.8
python -u experiments/run.py --model claude-opus-4-8 --provider anthropic --scenarios baseline --mazes 1 --n-runs 5

# Terminal 4 — GPT-5.5
python -u experiments/run.py --model gpt-5.5 --provider openai --scenarios baseline --mazes 1 --n-runs 5
```

Supported models (pass as `--model`):

| Model | Provider | Input $/1M | Output $/1M | Cache hit $/1M |
|---|---|---|---|---|
| `claude-opus-4-8` | `anthropic` | $5.00 | $25.00 | $0.50 |
| `claude-sonnet-4-6` | `anthropic` | $3.00 | $15.00 | $0.30 |
| `claude-haiku-4-5-20251001` | `anthropic` | $1.00 | $5.00 | $0.10 |
| `gpt-5.5` | `openai` | $5.00 | $30.00 | $0.50 |
| `gpt-5.4` | `openai` | $2.50 | $15.00 | $0.25 |
| `gpt-5` | `openai` | $1.25 | $10.00 | $0.125 |
| `gpt-4o` | `openai` | $2.50 | $10.00 | $1.25 |
| `gpt-4o-mini` | `openai` | $0.15 | $0.60 | $0.075 |
| `gemini/gemini-3.1-pro-preview` | `google` | $2.00 | $12.00 | $0.40 |
| `gemini/gemini-2.0-flash` | `google` | $0.075 | $0.30 | — |
| `deepseek/deepseek-v4-pro` | `deepseek` | $0.435 | $0.87 | $0.0043 |
| `deepseek/deepseek-v4-flash` | `deepseek` | $0.14 | $0.28 | $0.0028 |
| `deepseek/deepseek-chat` | `deepseek` | $0.14 | $0.28 | $0.0028 |
| `deepseek/deepseek-v3` | `deepseek` | $0.14 | $0.28 | $0.0028 |
| `deepseek/deepseek-v3.2` | `deepseek` | $0.14 | $0.28 | $0.0028 |

### Running locally with Ollama (free)

[Ollama](https://ollama.com) lets you run models on your own machine at no cost. Models must support tool/function calling.

1. Install Ollama and pull a model:
   ```bash
   ollama pull ZimaBlueAI/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF
   ```

2. Run with `--api-base` pointing to the local Ollama server and `openai/` prefix on the model name:
   ```bash
   python -u experiments/run.py --scenarios baseline --mazes 1 --n-runs 1 --model openai/<model-name> --provider ollama --api-base http://localhost:11434/v1 --api-key ollama --live --no-db
   ```

Tested local models:

| Model (Ollama name) | VRAM | Tool calling |
|---|---|---|
| `ZimaBlueAI/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF` | ~6 GB | Good |

Key flags:

| Flag | Description |
|---|---|
| `--model` | LiteLLM model string (e.g. `gpt-4o`, `claude-sonnet-4-6`) |
| `--provider` | Provider name (`openai`, `anthropic`, `ollama`) |
| `--api-base` | Custom API base URL (e.g. `http://localhost:11434/v1` for Ollama) |
| `--api-key` | Custom API key (use `ollama` for local Ollama) |
| `--scenarios` | One or more of `baseline`, `shared_memory`, `observer`, `shared_memory_observer` |
| `--mazes` | Maze IDs to run (default: all) |
| `--n-runs` | Runs per scenario/maze combination (default: 20) |
| `--live` | Show real-time maze visualization |
| `--no-db` | Skip Supabase, save results locally to `results/` |
| `--no-video` | Skip GIF video generation |

## Saving results to Supabase

Runs are always saved locally before any DB write:
```
results/
  experiments/     ← one JSON per run (source of truth for sync)
  shared_memory/   ← one JSONL per run, written incrementally during execution
  videos/
    <timestamp>/   ← one GIF per run, grouped by batch (pass --no-video to skip)
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
