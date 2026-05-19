import asyncio
from datetime import datetime, timezone
from src.maze.generator import Maze
from src.memory.shared import SharedMemoryStore
from src.agents.navigator import NavigatorAgent
from src.agents.observer import ObserverAgent
from src.scenarios.config import ModelConfig

SCENARIOS = ("baseline", "shared_memory", "shared_memory_observer")


async def run_scenario(
    scenario: str,
    maze: Maze,
    model_config: ModelConfig,
    run_number: int,
    lookahead: int = 3,
    on_move=None,
) -> dict:
    assert scenario in SCENARIOS, f"Unknown scenario: {scenario}"

    shared_memory = SharedMemoryStore() if scenario != "baseline" else None

    observer = (
        ObserverAgent(maze, model_config.model, shared_memory)
        if scenario == "shared_memory_observer"
        else None
    )

    agents = [
        NavigatorAgent(
            agent_id=str(i + 1),
            agent_index=i,
            maze=maze,
            model=model_config.model,
            lookahead=lookahead,
            shared_memory=shared_memory,
            observer=observer,
            on_move=on_move,
        )
        for i in range(3)
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    agent_results = await asyncio.gather(*[a.run() for a in agents])
    completed_at = datetime.now(timezone.utc).isoformat()

    observer_tokens = None
    if observer:
        observer_tokens = {
            "prompt_tokens": observer.prompt_tokens,
            "completion_tokens": observer.completion_tokens,
            "total_tokens": observer.prompt_tokens + observer.completion_tokens,
        }

    total_prompt = sum(r["prompt_tokens"] for r in agent_results)
    total_completion = sum(r["completion_tokens"] for r in agent_results)
    if observer_tokens:
        total_prompt += observer_tokens["prompt_tokens"]
        total_completion += observer_tokens["completion_tokens"]

    return {
        "scenario": scenario,
        "maze_id": None,       # caller fills this in
        "maze_seed": maze.seed,
        "provider": model_config.provider,
        "model": model_config.model,
        "model_version": model_config.version,
        "run_number": run_number,
        "lookahead": lookahead,
        "started_at": started_at,
        "completed_at": completed_at,
        "agents": list(agent_results),
        "observer_tokens": observer_tokens,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }