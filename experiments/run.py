import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

from src.maze.instances import get_maze, available_maze_ids
from src.scenarios.runner import run_scenario, SCENARIOS
from src.scenarios.config import ModelConfig
from src.db.client import SupabaseClient
from src.viz.terminal import LiveMazeView

RESULTS_DIR = Path("results")


async def run_experiments(
    model_config: ModelConfig,
    scenarios: list[str] = list(SCENARIOS),
    maze_ids: list[int] | None = None,
    n_runs: int = 20,
    lookahead: int = 3,
    save_to_db: bool = True,
    live: bool = False,
) -> None:
    if maze_ids is None:
        maze_ids = available_maze_ids()

    RESULTS_DIR.mkdir(exist_ok=True)

    db = await SupabaseClient.connect() if save_to_db else None

    total = len(scenarios) * len(maze_ids) * n_runs
    completed = 0
    failed = 0

    print(f"Starting {total} runs — model: {model_config.model}")
    print(f"Scenarios: {scenarios}")
    print(f"Mazes: {maze_ids}  |  Runs per combo: {n_runs}\n")

    for scenario in scenarios:
        for maze_id in maze_ids:
            maze = get_maze(maze_id)

            for run_number in range(1, n_runs + 1):
                completed += 1
                label = f"[{completed}/{total}] {scenario} | maze={maze_id} | run={run_number}"
                print(label, end=" ... ", flush=True)

                try:
                    on_move = None
                    if live:
                        view = LiveMazeView(maze, model=model_config.model)
                        on_move = view.update
                        print()  # space before first render

                    result = await run_scenario(
                        scenario=scenario,
                        maze=maze,
                        model_config=model_config,
                        run_number=run_number,
                        lookahead=lookahead,
                        on_move=on_move,
                    )
                    result["maze_id"] = maze_id

                    if live:
                        view.show_summary(result)

                    # Local JSON backup — always saved regardless of DB
                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    fname = f"{scenario}_maze{maze_id}_run{run_number:02d}_{ts}.json"
                    (RESULTS_DIR / fname).write_text(json.dumps(result, default=str))

                    if db:
                        experiment_id = await db.save_run(result, maze_id)
                        print(f"saved ({experiment_id[:8]}…)")
                    else:
                        print("saved locally")

                except Exception as e:
                    failed += 1
                    print(f"FAILED — {e}")

    print(f"\nDone. {completed - failed}/{total} succeeded, {failed} failed.")
    if failed:
        print("Check results/ folder for locally saved runs.")
    await asyncio.sleep(0.25)  # allow SSL connections to drain before loop closes


def parse_args():
    parser = argparse.ArgumentParser(description="Run MazeMemory experiments")
    parser.add_argument("--model",    default="claude-sonnet-4-6", help="LiteLLM model string")
    parser.add_argument("--provider", default="anthropic",          help="Provider name")
    parser.add_argument("--version",  default=None,                 help="Model version label (defaults to model)")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--mazes",    nargs="+", type=int, default=None)
    parser.add_argument("--n-runs",   type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--no-db",   action="store_true", help="Skip Supabase, save locally only")
    parser.add_argument("--live",    action="store_true", help="Show live maze visualization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_config = ModelConfig(
        provider=args.provider,
        model=args.model,
        version=args.version or args.model,
    )
    asyncio.run(run_experiments(
        model_config=model_config,
        scenarios=args.scenarios,
        maze_ids=args.mazes,
        n_runs=args.n_runs,
        lookahead=args.lookahead,
        save_to_db=not args.no_db,
        live=args.live,
    ))