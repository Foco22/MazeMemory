from pathlib import Path
from src.maze.generator import generate_maze
from src.viz.video import render_run_video, render_live_frame


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


def test_render_live_frame(tmp_path):
    maze = generate_maze(seed=42, rows=11, cols=11)
    positions = {"1": maze.start_positions[0], "2": maze.start_positions[1]}
    out = tmp_path / "frame.png"
    render_live_frame(maze, positions, out, title="t=3")
    assert out.exists(), "PNG frame was not created"
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "File does not have PNG magic bytes"


def test_render_live_frame_overwrites_atomically(tmp_path):
    maze = generate_maze(seed=42, rows=11, cols=11)
    out = tmp_path / "frame.png"
    render_live_frame(maze, {"1": maze.start_positions[0]}, out)
    first_bytes = out.read_bytes()
    render_live_frame(maze, {"1": maze.start_positions[1]}, out)
    second_bytes = out.read_bytes()
    assert list(tmp_path.glob("*.tmp")) == [], "Temp file was left behind"
    assert first_bytes != second_bytes, "Frame did not update for a new agent position"


def test_render_live_frame_tmp_name_is_unique_per_call(tmp_path, monkeypatch):
    # Two overlapping writers (e.g. a stale run.py process left over from a
    # previous Streamlit session) must never share a tmp filename, or one's
    # rename can race out from under the other and crash the run.
    maze = generate_maze(seed=42, rows=11, cols=11)
    out = tmp_path / "frame.png"
    seen_tmp_names = []

    import pathlib
    real_replace = pathlib.Path.replace

    def spying_replace(self, target):
        seen_tmp_names.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", spying_replace)

    render_live_frame(maze, {"1": maze.start_positions[0]}, out)
    render_live_frame(maze, {"1": maze.start_positions[1]}, out)

    assert len(seen_tmp_names) == 2
    assert seen_tmp_names[0] != seen_tmp_names[1], "tmp filename was reused across calls"
