import pytest
from src.maze.generator import generate_maze
from src.maze.pathfinding import astar
from src.metrics.calculator import PathOptimalityRatio, TokenConsumption


@pytest.fixture
def maze():
    return generate_maze(seed=42)


def make_agent_result(agent_id, path, reached_exit=True):
    return {
        "agent_id": agent_id,
        "path": path,
        "steps": len(path) - 1,
        "reached_exit": reached_exit,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "trace": [],
    }


class TestPathOptimalityRatio:
    def test_ratio_is_one_on_optimal_path(self, maze):
        optimal_path = astar(maze, maze.start_positions[0], maze.exit_pos)
        agent = make_agent_result("1", optimal_path)
        result = PathOptimalityRatio(maze).compute(agent)

        assert result["ratio"] == 1.0
        assert result["actual_steps"] == result["optimal_steps"]

    def test_ratio_above_one_for_suboptimal_path(self, maze):
        optimal_path = astar(maze, maze.start_positions[0], maze.exit_pos)
        detour = [optimal_path[0], optimal_path[1], optimal_path[0]] + optimal_path[1:]
        agent = make_agent_result("1", detour)
        result = PathOptimalityRatio(maze).compute(agent)

        assert result["ratio"] > 1.0

    def test_ratio_is_none_when_exit_not_reached(self, maze):
        start = maze.start_positions[0]
        agent = make_agent_result("1", [start, maze.neighbors(*start)[0]], reached_exit=False)
        result = PathOptimalityRatio(maze).compute(agent)

        assert result["ratio"] is None

    def test_compute_all_returns_one_per_agent(self, maze):
        agents = [
            make_agent_result(str(i + 1), astar(maze, maze.start_positions[i], maze.exit_pos))
            for i in range(3)
        ]
        results = PathOptimalityRatio(maze).compute_all({"agents": agents})

        assert len(results) == 3
        assert all(r["ratio"] == 1.0 for r in results)


class TestTokenConsumption:
    def make_run_result(self, prompt=300, completion=150, observer_tokens=None):
        return {
            "model": "claude-sonnet-4-6",
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_tokens": prompt + completion,
            "observer_tokens": observer_tokens,
            "agents": [
                {
                    "agent_id": str(i + 1),
                    "prompt_tokens": prompt // 3,
                    "completion_tokens": completion // 3,
                    "total_tokens": (prompt + completion) // 3,
                }
                for i in range(3)
            ],
        }

    def test_total_tokens_aggregated(self):
        result = TokenConsumption().compute(self.make_run_result(300, 150))
        assert result["prompt_tokens"] == 300
        assert result["completion_tokens"] == 150
        assert result["total_tokens"] == 450

    def test_cost_calculated(self):
        result = TokenConsumption().compute(self.make_run_result(1_000_000, 1_000_000))
        # claude-sonnet-4-6: $3/M input + $15/M output = $18
        assert result["estimated_cost_usd"] == 18.0

    def test_unknown_model_cost_is_zero(self):
        run = self.make_run_result()
        run["model"] = "unknown-model"
        result = TokenConsumption().compute(run)
        assert result["estimated_cost_usd"] == 0.0

    def test_per_agent_breakdown(self):
        result = TokenConsumption().compute(self.make_run_result())
        assert len(result["per_agent"]) == 3
        assert all("prompt_tokens" in a for a in result["per_agent"])

    def test_observer_tokens_included(self):
        obs = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        result = TokenConsumption().compute(self.make_run_result(observer_tokens=obs))
        assert result["observer_tokens"]["total_tokens"] == 70