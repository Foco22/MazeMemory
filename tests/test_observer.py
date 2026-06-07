import sys
import pytest
from unittest.mock import MagicMock

# litellm is not installed in the test environment; mock it before importing observer
sys.modules.setdefault("litellm", MagicMock())

from src.maze.generator import generate_maze
from src.agents.observer import ObserverAgent


@pytest.fixture
def maze():
    return generate_maze(seed=42)


@pytest.fixture
def observer(maze):
    shared_memory = MagicMock()
    return ObserverAgent(maze=maze, model="test-model", shared_memory=shared_memory)


def make_trajectories(agents: dict[str, list[tuple[int, int, int]]]) -> dict:
    return agents


HEADER_ROWS = 2   # x-axis label rows prepended by _render_with_agents
X_OFFSET = 4      # y-axis label prefix: "DD  " (2 digits + 2 spaces)


def cell(lines, x, y):
    """Return the character at maze coordinate (x, y) accounting for header/prefix."""
    return lines[y + HEADER_ROWS][x + X_OFFSET]


class TestRenderWithAgents:
    def test_exit_always_shown(self, maze, observer):
        rendered = observer._render_with_agents({})
        lines = rendered.splitlines()
        ex, ey = maze.exit_pos
        assert cell(lines, ex, ey) == "E"

    def test_walls_shown_when_no_trajectories(self, maze, observer):
        rendered = observer._render_with_agents({})
        lines = rendered.splitlines()
        # top-left corner (0,0) is always a wall in this maze generator
        assert cell(lines, 0, 0) == "█"

    def test_explored_cells_marked(self, maze, observer):
        # Agent 1 visits (1,1) then moves to (2,1), so (1,1) shows ·
        trajectories = {"1": [(1, 1, 0), (2, 1, 1)]}
        rendered = observer._render_with_agents(trajectories)
        lines = rendered.splitlines()
        assert cell(lines, 1, 1) == "·"

    def test_current_agent_position_shown(self, maze, observer):
        trajectories = {"1": [(1, 1, 0), (2, 1, 1)]}
        rendered = observer._render_with_agents(trajectories)
        lines = rendered.splitlines()
        # Agent 1 is now at (2,1)
        assert cell(lines, 2, 1) == "1"

    def test_empty_trajectories_shows_plain_maze(self, maze, observer):
        rendered = observer._render_with_agents({})
        assert "·" not in rendered

    def test_multiple_agents_shown(self, maze, observer):
        trajectories = {
            "1": [(1, 1, 0)],
            "2": [(1, 13, 0)],
        }
        rendered = observer._render_with_agents(trajectories)
        lines = rendered.splitlines()
        assert cell(lines, 1, 1) == "1"
        assert cell(lines, 1, 13) == "2"


class TestComputeAgentStatuses:
    def test_returns_one_entry_per_agent(self, maze, observer):
        trajectories = {
            "1": [(1, 1, 0), (2, 1, 1)],
            "2": [(1, 13, 0)],
            "3": [(13, 1, 0)],
        }
        statuses = observer._compute_agent_statuses(trajectories)
        assert len(statuses) == 3

    def test_sorted_by_agent_id(self, maze, observer):
        trajectories = {"3": [(13, 1, 0)], "1": [(1, 1, 0)], "2": [(1, 13, 0)]}
        statuses = observer._compute_agent_statuses(trajectories)
        assert [s["agent_id"] for s in statuses] == ["1", "2", "3"]

    def test_steps_taken_matches_trajectory_length(self, maze, observer):
        trajectories = {"1": [(1, 1, 0), (2, 1, 1), (3, 1, 2)]}
        statuses = observer._compute_agent_statuses(trajectories)
        assert statuses[0]["steps_taken"] == 3

    def test_position_is_last_recorded(self, maze, observer):
        trajectories = {"1": [(1, 1, 0), (2, 1, 1), (3, 1, 2)]}
        statuses = observer._compute_agent_statuses(trajectories)
        assert statuses[0]["position"] == (3, 1)

    def test_distance_to_exit_is_positive(self, maze, observer):
        trajectories = {"2": [(1, 13, 0)]}
        statuses = observer._compute_agent_statuses(trajectories)
        assert statuses[0]["distance_to_exit"] > 0

    def test_distance_zero_at_exit(self, maze, observer):
        ex, ey = maze.exit_pos
        trajectories = {"1": [(ex, ey, 0)]}
        statuses = observer._compute_agent_statuses(trajectories)
        assert statuses[0]["distance_to_exit"] == 0

    def test_empty_agent_steps_skipped(self, maze, observer):
        trajectories = {"1": [], "2": [(1, 13, 0)]}
        statuses = observer._compute_agent_statuses(trajectories)
        assert len(statuses) == 1
        assert statuses[0]["agent_id"] == "2"


class TestNearestWaypointOnPath:
    def test_returns_closest_by_manhattan(self, observer):
        trajectory = [(1, 1, 0), (5, 5, 1), (10, 10, 2)]
        wp = observer._nearest_waypoint_on_path((4, 4), trajectory)
        assert wp == (5, 5)

    def test_returns_none_for_empty_trajectory(self, observer):
        assert observer._nearest_waypoint_on_path((3, 3), []) is None

    def test_returns_exact_match(self, observer):
        trajectory = [(2, 3, 0), (7, 8, 1)]
        wp = observer._nearest_waypoint_on_path((7, 8), trajectory)
        assert wp == (7, 8)

    def test_single_cell_trajectory(self, observer):
        trajectory = [(5, 5, 0)]
        wp = observer._nearest_waypoint_on_path((1, 1), trajectory)
        assert wp == (5, 5)