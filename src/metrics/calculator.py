import json
from pathlib import Path
from src.maze.generator import Maze
from src.maze.pathfinding import optimal_steps

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

        prompt     = run_result["total_prompt_tokens"]
        completion = run_result["total_completion_tokens"]
        total      = run_result["total_tokens"]

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
        }