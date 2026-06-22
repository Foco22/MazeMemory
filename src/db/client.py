import os
from datetime import datetime, timezone
from supabase._async.client import AsyncClient, create_client
from dotenv import load_dotenv
from src.metrics.calculator import TokenConsumption

load_dotenv()


class SupabaseClient:
    def __init__(self, client: AsyncClient):
        self._client = client

    @classmethod
    async def connect(cls) -> "SupabaseClient":
        client = await create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
        return cls(client)

    async def run_exists(self, result: dict) -> bool:
        resp = (
            await self._client.table("experiments")
            .select("id")
            .eq("scenario",   result["scenario"])
            .eq("maze_id",    result["maze_id"])
            .eq("started_at", result["started_at"])
            .execute()
        )
        return len(resp.data) > 0

    async def save_run(self, result: dict, maze_id: int, batch_id: str | None = None) -> str:
        tokens = TokenConsumption().compute(result)
        price = tokens["_price"]
        price_in  = price["input"]
        price_out = price["output"]

        cost_completion = round(result["total_completion_tokens"] * price_out / 1_000_000, 6)
        cost_prompt     = round(tokens["estimated_cost_usd"] - cost_completion, 6)
        cost_agents     = round(tokens["estimated_cost_usd"], 6)

        obs = result["observer_tokens"]
        cost_observer = round(
            (obs["prompt_tokens"] * price_in + obs["completion_tokens"] * price_out) / 1_000_000, 6
        ) if obs else None

        cost_total = round(cost_agents + (cost_observer or 0), 6)

        started  = datetime.fromisoformat(result["started_at"])
        completed = datetime.fromisoformat(result["completed_at"])
        exp_duration = round((completed - started).total_seconds(), 3)

        exp_row = {
            "batch_id":                   batch_id,
            "scenario":                   result["scenario"],
            "maze_id":                    maze_id,
            "maze_seed":                  result["maze_seed"],
            "provider":                   result["provider"],
            "model":                      result["model"],
            "model_version":              result["model_version"],
            "run_number":                 result["run_number"],
            "lookahead":                  result["lookahead"],
            "started_at":                 result["started_at"],
            "completed_at":               result["completed_at"],
            "total_prompt_tokens":        result["total_prompt_tokens"],
            "total_completion_tokens":    result["total_completion_tokens"],
            "total_tokens":               result["total_tokens"],
            "total_cache_hit_tokens":     result.get("total_cache_hit_tokens"),
            "total_cache_miss_tokens":    result.get("total_cache_miss_tokens"),
            "observer_prompt_tokens":      obs["prompt_tokens"]      if obs else None,
            "observer_completion_tokens":  obs["completion_tokens"]  if obs else None,
            "observer_cache_hit_tokens":   obs.get("cache_hit_tokens")  if obs else None,
            "observer_cache_miss_tokens":  obs.get("cache_miss_tokens") if obs else None,
            "cost_prompt_usd":            cost_prompt,
            "cost_completion_usd":        cost_completion,
            "cost_agents_usd":            cost_agents,
            "cost_observer_usd":          cost_observer,
            "cost_total_usd":             cost_total,
            "duration_seconds":           exp_duration,
        }

        exp_response = await self._client.table("experiments").insert(exp_row).execute()
        experiment_id = exp_response.data[0]["id"]

        agent_rows = [
            {
                "experiment_id":    experiment_id,
                "agent_id":         a["agent_id"],
                "steps":            a["steps"],
                "optimal_steps":    a.get("optimal_steps"),
                "optimality_ratio": a.get("optimality_ratio"),
                "prompt_tokens":    a["prompt_tokens"],
                "completion_tokens": a["completion_tokens"],
                "total_tokens":     a["total_tokens"],
                "cache_hit_tokens":   a.get("cache_hit_tokens"),
                "cache_miss_tokens":  a.get("cache_miss_tokens"),
                "reached_exit":       a["reached_exit"],
                "duration_seconds":   a.get("duration_seconds"),
                "redundant_cells":    a.get("redundant_cells"),
                "total_cells_visited": a.get("total_cells_visited"),
                "redundancy_ratio":   a.get("redundancy_ratio"),
            }
            for a in result["agents"]
        ]
        await self._client.table("agent_runs").insert(agent_rows).execute()

        trajectory_rows = [
            {
                "experiment_id": experiment_id,
                "agent_id":      a["agent_id"],
                "x":             x,
                "y":             y,
                "timestep":      t,
            }
            for a in result["agents"]
            for t, (x, y) in enumerate(a["path"])
        ]
        await self._client.table("trajectories").insert(trajectory_rows).execute()

        action_rows = [
            {
                "experiment_id":    experiment_id,
                "agent_id":         a["agent_id"],
                "step":             action["step"],
                "llm_text":         action["llm_text"],
                "tool_name":        action["tool_name"],
                "tool_args":        action["tool_args"],
                "tool_result":      action["tool_result"],
                "prompt_tokens":    action.get("prompt_tokens"),
                "completion_tokens": action.get("completion_tokens"),
            }
            for a in result["agents"]
            for action in a.get("trace", [])
        ]
        if action_rows:
            await self._client.table("agent_actions").insert(action_rows).execute()

        return experiment_id