import json
from pathlib import Path
from src.maze.generator import Maze
from src.maze.pathfinding import optimal_steps, astar

_PRICES_PATH = Path(__file__).parent / "prices.json"
with _PRICES_PATH.open() as f:
    _PRICES: dict[str, dict] = json.load(f)


class PathOptimalityRatio:
    def __init__(self, maze: Maze):
        self.maze = maze

    def compute(self, agent_result: dict) -> dict:
        agent_index = int(agent_result["agent_id"]) - 1
        start = self.maze.start_positions[agent_index]
        opt = optimal_steps(self.maze, start, self.maze.exit_pos)

        if not opt or not agent_result["reached_exit"]:
            return {
                "agent_id": agent_result["agent_id"],
                "actual_steps": agent_result["steps"],
                "optimal_steps": opt,
                "ratio": None,
            }

        return {
            "agent_id": agent_result["agent_id"],
            "actual_steps": agent_result["steps"],
            "optimal_steps": opt,
            "ratio": agent_result["steps"] / opt,
        }

    def compute_all(self, run_result: dict) -> list[dict]:
        return [self.compute(a) for a in run_result["agents"]]


class TokenConsumption:
    def compute(self, run_result: dict) -> dict:
        model = run_result["model"]
        price = _PRICES.get(model, {"input": 0.0, "output": 0.0})
        price_in, price_out = price["input"], price["output"]
        price_cache = price.get("cache_hit")

        prompt     = run_result["total_prompt_tokens"]
        completion = run_result["total_completion_tokens"]
        total      = run_result["total_tokens"]

        cache_hit  = run_result.get("total_cache_hit_tokens")
        cache_miss = run_result.get("total_cache_miss_tokens")

        if price_cache is not None and cache_hit is not None and cache_miss is not None:
            uncategorized = max(0, prompt - cache_hit - cache_miss)
            cost_usd = (cache_hit * price_cache + (cache_miss + uncategorized) * price_in + completion * price_out) / 1_000_000
        else:
            cost_usd = (prompt * price_in + completion * price_out) / 1_000_000

        per_agent = [
            {
                "agent_id":         a["agent_id"],
                "prompt_tokens":    a["prompt_tokens"],
                "completion_tokens": a["completion_tokens"],
                "total_tokens":     a["total_tokens"],
            }
            for a in run_result["agents"]
        ]

        observer = run_result.get("observer_tokens")

        return {
            "model":                    model,
            "prompt_tokens":            prompt,
            "completion_tokens":        completion,
            "total_tokens":             total,
            "estimated_cost_usd":       round(cost_usd, 6),
            "observer_tokens":          observer,
            "per_agent":                per_agent,
            "_price":                   {"input": price_in, "output": price_out},
        }


class RedundantComputationReduction:
    """
    Measures how much work agents duplicate after the first exit is found.

    For each agent i:
      redundant_cells = cells visited after T_first_exit that were already
                        visited by another agent AND are not on agent i's
                        optimal A* path from its position at T_first_exit.
      ratio = redundant_cells / total_unique_cells_visited_by_agent_i

    Exploration before T_first_exit is never penalised.
    """

    def __init__(self, maze: Maze):
        self.maze = maze

    def compute(self, run_result: dict) -> list[dict]:
        agents = run_result["agents"]

        exit_steps = [a["steps"] for a in agents if a["reached_exit"]]
        if not exit_steps:
            return [
                {
                    "agent_id": a["agent_id"],
                    "redundant_cells": 0,
                    "total_cells_visited": len({tuple(p) for p in a["path"]}),
                    "ratio": None,
                    "t_first_exit": None,
                }
                for a in agents
            ]

        t_first_exit = min(exit_steps)
        results = []

        for agent in agents:
            path = [tuple(p) for p in agent["path"]]
            agent_id = agent["agent_id"]

            other_cells: set[tuple] = set()
            for other in agents:
                if other["agent_id"] != agent_id:
                    other_cells.update(tuple(p) for p in other["path"])

            idx = min(t_first_exit, len(path) - 1)
            pos_at_first_exit = path[idx]

            optimal_set = set(astar(self.maze, pos_at_first_exit, self.maze.exit_pos))

            post_exit = path[t_first_exit + 1:]
            redundant = {c for c in post_exit if c in other_cells and c not in optimal_set}

            total_cells = len(set(path))
            n_redundant = len(redundant)

            results.append({
                "agent_id": agent_id,
                "redundant_cells": n_redundant,
                "total_cells_visited": total_cells,
                "ratio": round(n_redundant / total_cells, 4) if total_cells > 0 else None,
                "t_first_exit": t_first_exit,
                "pos_at_first_exit": list(pos_at_first_exit),
            })

        return results