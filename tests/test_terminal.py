import pytest
from unittest.mock import patch
from src.maze.generator import generate_maze
from src.maze.pathfinding import astar
from src.viz.terminal import LiveMazeView


@pytest.fixture
def maze():
    return generate_maze(seed=42, rows=15, cols=15)


def make_run_result(maze, agents_override=None):
    agents = agents_override or [
        {
            "agent_id": str(i + 1),
            "path": astar(maze, maze.start_positions[i], maze.exit_pos),
            "steps": len(astar(maze, maze.start_positions[i], maze.exit_pos)) - 1,
            "reached_exit": True,
            "prompt_tokens": 100 * (i + 1),
            "completion_tokens": 20 * (i + 1),
            "total_tokens": 120 * (i + 1),
            "trace": [],
        }
        for i in range(3)
    ]
    total_prompt = sum(a["prompt_tokens"] for a in agents)
    total_completion = sum(a["completion_tokens"] for a in agents)
    return {
        "scenario": "baseline",
        "maze_id": 1,
        "model": "gpt-4o-mini",
        "agents": agents,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "observer_tokens": None,
    }


@pytest.fixture
def view(maze):
    with patch("src.viz.terminal._is_tty", return_value=True), \
         patch("src.viz.terminal._enter_alt_screen"), \
         patch("src.viz.terminal.atexit.register"):
        yield LiveMazeView(maze, model="gpt-4o-mini")


class TestShowSummary:
    def test_prints_scenario_and_maze_id(self, view, maze, capsys):
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen"), \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        out = capsys.readouterr().out
        assert "baseline" in out
        assert "maze=1" in out

    def test_prints_all_three_agents(self, view, maze, capsys):
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen"), \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        out = capsys.readouterr().out
        assert "Agent 1" in out
        assert "Agent 2" in out
        assert "Agent 3" in out

    def test_optimal_ratio_is_1_for_optimal_path(self, view, maze, capsys):
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen"), \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        out = capsys.readouterr().out
        assert "1.00" in out

    def test_ratio_na_when_exit_not_reached(self, view, maze, capsys):
        start = maze.start_positions[0]
        agent_dnf = {
            "agent_id": "1",
            "path": [start],
            "steps": 0,
            "reached_exit": False,
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "trace": [],
        }
        other_agents = [
            {
                "agent_id": str(i + 1),
                "path": astar(maze, maze.start_positions[i], maze.exit_pos),
                "steps": len(astar(maze, maze.start_positions[i], maze.exit_pos)) - 1,
                "reached_exit": True,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "trace": [],
            }
            for i in range(1, 3)
        ]
        result = make_run_result(maze, agents_override=[agent_dnf] + other_agents)
        with patch("src.viz.terminal._exit_alt_screen"), \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        out = capsys.readouterr().out
        assert "N/A" in out

    def test_prints_total_tokens(self, view, maze, capsys):
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen"), \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        out = capsys.readouterr().out
        assert str(result["total_tokens"]) in out

    def test_prints_cost(self, view, maze, capsys):
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen"), \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        out = capsys.readouterr().out
        # total cost row + one cost column entry per agent (4 dollar signs total)
        assert out.count("$") >= 4

    def test_exits_alt_screen(self, view, maze):
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen") as mock_exit, \
             patch("src.viz.terminal.atexit.unregister"):
            view.show_summary(result)
        mock_exit.assert_called_once()


class TestNonTtyOutput:
    """When stdout is piped (e.g. Streamlit's subprocess), no raw ANSI
    escape codes should be written — they'd show up as garbage text instead
    of being interpreted, since there's no real terminal to interpret them.
    """

    @pytest.fixture
    def piped_view(self, maze, tmp_path):
        with patch("src.viz.terminal._is_tty", return_value=False):
            yield LiveMazeView(maze, model="gpt-4o-mini", frame_path=tmp_path / "frame.png")

    def test_does_not_enter_alt_screen(self, maze):
        with patch("src.viz.terminal._is_tty", return_value=False), \
             patch("src.viz.terminal._enter_alt_screen") as mock_enter:
            LiveMazeView(maze, model="gpt-4o-mini")
        mock_enter.assert_not_called()

    def test_render_has_no_escape_codes(self, piped_view, maze, capsys):
        x, y = maze.start_positions[0]
        import asyncio
        asyncio.run(piped_view.update("1", (x, y), timestep=0))
        out = capsys.readouterr().out
        assert "\033" not in out
        assert "█" not in out, "ASCII maze grid should not be printed when piped"

    def test_render_writes_png_frame_instead_of_ascii(self, piped_view, maze):
        x, y = maze.start_positions[0]
        import asyncio
        asyncio.run(piped_view.update("1", (x, y), timestep=0))
        assert piped_view.frame_path.exists()
        assert piped_view.frame_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_frame_write_failure_does_not_crash_the_run(self, piped_view, maze):
        # A transient OSError writing the live-view PNG (e.g. a race with a
        # stale writer on the same tmp path) must not abort the agent run —
        # this is a best-effort visualization, not a correctness requirement.
        x, y = maze.start_positions[0]
        import asyncio
        with patch("src.viz.terminal.render_live_frame", side_effect=OSError("No such file or directory")):
            asyncio.run(piped_view.update("1", (x, y), timestep=0))  # should not raise

    def test_show_summary_is_a_no_op(self, piped_view, maze, capsys):
        # The RUN SUMMARY table is shown in Streamlit's Results section
        # instead — printing it here too would just duplicate it in the log.
        result = make_run_result(maze)
        with patch("src.viz.terminal._exit_alt_screen") as mock_exit, \
             patch("src.viz.terminal.atexit.unregister") as mock_unregister:
            piped_view.show_summary(result)
        out = capsys.readouterr().out
        assert out == ""
        mock_exit.assert_not_called()
        mock_unregister.assert_not_called()
