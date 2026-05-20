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
    with patch("src.viz.terminal._enter_alt_screen"), \
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
