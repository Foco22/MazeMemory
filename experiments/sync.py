"""
Upload locally-saved run JSONs from results/ to Supabase.
Skips runs already in the database (deduplicates by scenario + maze_id + started_at).

Usage:
    python experiments/sync.py
    python experiments/sync.py --results-dir path/to/results
    python experiments/sync.py --dry-run
"""
import asyncio
import argparse
import json
from pathlib import Path

from src.db.client import SupabaseClient

RESULTS_DIR = Path("results")


async def sync(results_dir: Path, dry_run: bool = False) -> None:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {results_dir}/")
        return

    db = await SupabaseClient.connect()

    uploaded = skipped = failed = 0

    for path in files:
        result = json.loads(path.read_text())
        label = path.name

        try:
            if await db.run_exists(result):
                print(f"  skip  {label}  (already in DB)")
                skipped += 1
                continue

            if dry_run:
                print(f"  would upload  {label}")
                uploaded += 1
                continue

            experiment_id = await db.save_run(result, result["maze_id"])
            print(f"  ok    {label}  → {experiment_id[:8]}…")
            uploaded += 1

        except Exception as e:
            print(f"  FAIL  {label}  — {e}")
            failed += 1

    action = "would upload" if dry_run else "uploaded"
    print(f"\nDone. {action}={uploaded}  skipped={skipped}  failed={failed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sync local results to Supabase")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without writing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(sync(args.results_dir, dry_run=args.dry_run))
