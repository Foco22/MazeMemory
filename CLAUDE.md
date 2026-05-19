# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MazeMemory is a Master's thesis project (Birkbeck) that evaluates the effect of **shared memory in LLM-based multi-agent systems** using a maze navigation environment. The codebase is under active development; see `MazeMemory.md` for the full research design.

## Research Design

Three experimental scenarios:

| Scenario | Memory | Observer |
|---|---|---|
| Baseline | Private per-agent | No |
| Shared Memory | Shared in-memory store + private | No |
| Shared Memory + Observer | Shared in-memory store + private | Yes |

Each scenario runs **20 times × 3 maze instances = 60 runs per scenario** (180 total). Maze instances must be seeded so runs are reproducible across all three scenarios.

## Agent Architecture

Agents navigate a maze that is **unknown to them** — they have no global map; they can only query their current state via tools.

**Navigator agents** (3 per run) have two tools:
- `get_location()` — returns agent's current `(x, y)` position
- `get_distance_to_exit()` — returns steps-to-exit from current position

**Observer agent** (Shared Memory + Observer scenario only) has one tool:
- `get_insight()` — reconstructs all agents' paths from shared memory and returns a movement recommendation to the invoking agent

In Shared Memory scenarios, each agent writes `(agent_id, x, y, timestep)` to the shared store on **every move**, not just at the end of a run. This is what `get_insight()` reads to reconstruct trajectories.

## Key Metrics (must be tracked per run)

1. **Path Optimality Ratio** — `actual_steps / optimal_A*_steps` per agent; 1.0 = optimal, higher = more exploration overhead.

2. **Token Consumption** — total tokens across all agents in a run; compared against baseline to quantify cost reduction.

3. **Redundant Computation Reduction** — per agent: `redundant_cells / total_cells_visited`, where a cell is redundant if **both** conditions hold:
   - It was already visited by a different agent, AND
   - It is not on the agent's optimal A* path from its position **at the moment the first agent finds the exit**

   Exploration before the first agent finds the exit is not penalised — agents are navigating an unknown environment and free exploration is unavoidable up to that point.

## Workflow
- Ask clarifying questions before starting complex or ambiguous tasks
- Make minimal changes — do not refactor unrelated code
- Run tests after every change; fix failures before moving on
- Create separate commits per logical change
- When unsure between approaches, explain both and let me choose
- For Any new functions or class, you must do a test to validate if works.

## Implementation Notes

- A* path calculation is needed both for the optimality ratio and the redundancy metric — implement once and reuse.
- Results must be serialised per-run (JSON or CSV) before aggregation so re-analysis is possible without re-running experiments.
- The shared memory store is in-memory within a single run; it does not persist across runs.

## Stack
- Language: Python
- LLM: LiteLLM (provider-agnostic async interface — supports Anthropic, OpenAI, Gemini)
- Concurrency: asyncio — 3 navigator agents run with `asyncio.gather()`
- Shared memory: in-memory dict + `asyncio.Lock()` during a run (not persisted to DB)
- Database: Supabase (PostgreSQL) — stores experiment metadata and raw trajectories after each run

### Supabase schema
```
experiments  (id, scenario, maze_id, maze_seed, run_number, started_at, completed_at)
trajectories (id, experiment_id, agent_id, x, y, timestep)
```