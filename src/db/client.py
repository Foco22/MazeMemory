import os
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

    async def save_run(self, result: dict, maze_id: int) -> str:
        tokens = TokenConsumption().compute(result)
        price = tokens["_price"]

        cost_prompt     = round(result["total_prompt_tokens"]     * price["input"]  / 1_000_000, 6)
        cost_completion = round(result["total_completion_tokens"] * price["output"] / 1_000_000, 6)
        cost_agents     = round(cost_prompt + cost_completion, 6)

        obs = result["observer_tokens"]
        cost_observer = round(
            (obs["prompt_tokens"] * price["input"] + obs["completion_tokens"] * price["output"]) / 1_000_000, 6
        ) if obs else None

        cost_total = round(cost_agents + (cost_observer or 0), 6)

        exp_row = {
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
            "observer_prompt_tokens":     obs["prompt_tokens"]     if obs else None,
            "observer_completion_tokens": obs["completion_tokens"] if obs else None,
            "cost_prompt_usd":            cost_prompt,
            "cost_completion_usd":        cost_completion,
            "cost_agents_usd":            cost_agents,
            "cost_observer_usd":          cost_observer,
            "cost_total_usd":             cost_total,
        }

        exp_response = await self._client.table("experiments").insert(exp_row).execute()
        experiment_id = exp_response.data[0]["id"]

        agent_rows = [
            {
                "experiment_id":    experiment_id,
                "agent_id":         a["agent_id"],
                "steps":            a["steps"],
                "prompt_tokens":    a["prompt_tokens"],
                "completion_tokens": a["completion_tokens"],
                "total_tokens":     a["total_tokens"],
                "reached_exit":     a["reached_exit"],
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
                "experiment_id": experiment_id,
                "agent_id":      a["agent_id"],
                "step":          action["step"],
                "llm_text":      action["llm_text"],
                "tool_name":     action["tool_name"],
                "tool_args":     action["tool_args"],
                "tool_result":   action["tool_result"],
            }
            for a in result["agents"]
            for action in a.get("trace", [])
        ]
        if action_rows:
            await self._client.table("agent_actions").insert(action_rows).execute()

        return experiment_id