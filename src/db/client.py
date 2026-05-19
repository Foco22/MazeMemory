import os
from supabase._async.client import AsyncClient, create_client
from dotenv import load_dotenv

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

    async def save_run(self, result: dict, maze_id: int) -> str:
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
            "observer_prompt_tokens":     result["observer_tokens"]["prompt_tokens"] if result["observer_tokens"] else None,
            "observer_completion_tokens": result["observer_tokens"]["completion_tokens"] if result["observer_tokens"] else None,
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