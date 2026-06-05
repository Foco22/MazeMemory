import asyncio
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timezone

from src.maze.instances import get_maze, available_maze_ids
from src.scenarios.runner import run_scenario, SCENARIOS
from src.scenarios.config import ModelConfig
from src.db.client import SupabaseClient
from src.viz.terminal import LiveMazeView
from src.viz.video import render_run_video

RESULTS_DIR    = Path("results/experiments")
SHARED_MEM_DIR = Path("results/shared_memory")
VIDEOS_DIR     = Path("results/videos")


async def run_experiments(
    model_config: ModelConfig,
    scenarios: list[str] = list(SCENARIOS),
    maze_ids: list[int] | None = None,
    n_runs: int = 20,
    lookahead: int = 3,
    save_to_db: bool = True,
    live: bool = False,
    save_video: bool = True,
) -> None:
    if maze_ids is None:
        maze_ids = available_maze_ids()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_MEM_DIR.mkdir(parents=True, exist_ok=True)

    batch_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    videos_folder = VIDEOS_DIR / batch_ts
    if save_video:
        videos_folder.mkdir(parents=True, exist_ok=True)

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
                    on_llm_call = None
                    if live:
                        view = LiveMazeView(maze, model=model_config.model)
                        on_move = view.update
                        on_llm_call = view.on_llm_call
                        print()  # space before first render

                    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    with open("trace.log", "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"RUN START: scenario={scenario} maze={maze_id} run={run_number} model={model_config.model} ts={ts}\n")
                        f.write(f"{'='*60}\n")

                    sm_log = None
                    if scenario != "baseline":
                        sm_log = SHARED_MEM_DIR / f"{scenario}_maze{maze_id}_run{run_number:02d}_{ts}.jsonl"

                    result = await run_scenario(
                        scenario=scenario,
                        maze=maze,
                        model_config=model_config,
                        run_number=run_number,
                        lookahead=lookahead,
                        on_move=on_move,
                        on_llm_call=on_llm_call,
                        shared_memory_log=sm_log,
                    )
                    result["maze_id"] = maze_id

                    if live:
                        view.show_summary(result)

                    # Local JSON backup — always saved regardless of DB
                    fname = f"{scenario}_maze{maze_id}_run{run_number:02d}_{ts}.json"
                    (RESULTS_DIR / fname).write_text(json.dumps(result, default=str))

                    if save_video:
                        gif_name = f"{scenario}_maze{maze_id}_run{run_number:02d}.gif"
                        render_run_video(maze, result, videos_folder / gif_name)

                    if db:
                        experiment_id = await db.save_run(result, maze_id)
                        print(f"saved ({experiment_id[:8]}…)")
                    else:
                        print("saved locally")

                except Exception as e:
                    failed += 1
                    print(f"FAILED — {e}")
                    traceback.print_exc()

    print(f"\nDone. {completed - failed}/{total} succeeded, {failed} failed.")
    if failed:
        print("Check results/experiments/ for locally saved runs.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MazeMemory experiments")
    parser.add_argument("--model",    default="claude-sonnet-4-6", help="LiteLLM model string")
    parser.add_argument("--provider", default="anthropic",          help="Provider name")
    parser.add_argument("--version",  default=None,                 help="Model version label (defaults to model)")
    parser.add_argument("--api-base", default=None,                 help="Custom API base URL (e.g. for Ollama: http://localhost:11434/v1)")
    parser.add_argument("--api-key",  default=None,                 help="Custom API key (use 'ollama' for local Ollama)")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--mazes",    nargs="+", type=int, default=None)
    parser.add_argument("--n-runs",   type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--no-db",    action="store_true", help="Skip Supabase, save locally only")
    parser.add_argument("--live",     action="store_true", help="Show live maze visualization")
    parser.add_argument("--no-video", action="store_true", help="Skip GIF video generation")
    return parser.parse_args()


def _silence_ssl_shutdown(loop, context):
    msg = context.get("message", "")
    if "SSL" in msg or "Fatal" in msg:
        return
    loop.default_exception_handler(context)


if __name__ == "__main__":
    args = parse_args()
    model_config = ModelConfig(
        provider=args.provider,
        model=args.model,
        version=args.version or args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(_silence_ssl_shutdown)
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_experiments(
            model_config=model_config,
            scenarios=args.scenarios,
            maze_ids=args.mazes,
            n_runs=args.n_runs,
            lookahead=args.lookahead,
            save_to_db=not args.no_db,
            live=args.live,
            save_video=not args.no_video,
        ))
    finally:
        loop.close()