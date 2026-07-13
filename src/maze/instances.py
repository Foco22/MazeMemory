import json
from pathlib import Path
from .generator import Maze, generate_maze

_CONFIG_PATH = Path(__file__).parent / "instances.json"

with _CONFIG_PATH.open() as f:
    _INSTANCES: dict[str, dict] = json.load(f)


def available_maze_ids() -> list[int]:
    return sorted(int(k) for k in _INSTANCES)


def maze_id_by_difficulty(difficulty: str) -> int:
    for k, v in _INSTANCES.items():
        if v.get("difficulty") == difficulty:
            return int(k)
    raise ValueError(f"No maze found with difficulty '{difficulty}'")


def get_maze(maze_id: int) -> Maze:
    config = _INSTANCES[str(maze_id)]
    return generate_maze(config["seed"], config["rows"], config["cols"])