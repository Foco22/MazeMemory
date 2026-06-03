from pathlib import Path
from src.maze.generator import generate_maze
from src.viz.video import render_run_video


def _make_result(maze):
    return {
        "scenario": "baseline",
        "maze_id": 0,
        "run_number": 1,
        "agents": [
            {"agent_id": "1", "path": [maze.start_positions[0], (1, 3), (1, 5)]},
            {"agent_id": "2", "path": [maze.start_positions[1], (1, 5), (3, 5)]},
            {"agent_id": "3", "path": [maze.start_positions[2], (3, 1), (3, 3)]},
        ],
    }


def test_render_run_video(tmp_path):
    maze = generate_maze(seed=42, rows=11, cols=11)
    result = _make_result(maze)
    out = tmp_path / "test.gif"
    render_run_video(maze, result, out)
    assert out.exists(), "GIF file was not created"
    assert out.read_bytes()[:2] == b"GI", "File does not have GIF magic bytes"
