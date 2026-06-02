import pytest
from src.maze.generator import generate_maze
from src.maze.pathfinding import astar
from src.metrics.calculator import PathOptimalityRatio, TokenConsumption, RedundantComputationReduction


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


class TestRedundantComputationReduction:
    def make_run(self, agents):
        return {"agents": agents}

    def test_no_exit_reached_returns_none_ratio(self, maze):
        agents = [
            make_agent_result(str(i + 1), [maze.start_positions[i]], reached_exit=False)
            for i in range(3)
        ]
        results = RedundantComputationReduction(maze).compute(self.make_run(agents))
        assert all(r["ratio"] is None for r in results)
        assert all(r["t_first_exit"] is None for r in results)

    def test_first_exit_agent_has_zero_redundancy(self, maze):
        # Agent 2's optimal path (16 steps) is the shortest — it exits first.
        paths = [astar(maze, maze.start_positions[i], maze.exit_pos) for i in range(3)]
        agents = [make_agent_result(str(i + 1), paths[i]) for i in range(3)]
        results = RedundantComputationReduction(maze).compute(self.make_run(agents))

        a2 = next(r for r in results if r["agent_id"] == "2")
        assert a2["redundant_cells"] == 0
        assert a2["ratio"] == 0.0

    def test_redundancy_matches_manual_computation(self, maze):
        """Verify the formula: post-exit cells ∩ other_agents − optimal_set."""
        paths = [astar(maze, maze.start_positions[i], maze.exit_pos) for i in range(3)]
        agents = [make_agent_result(str(i + 1), paths[i]) for i in range(3)]
        t = min(a["steps"] for a in agents)  # 16

        # Manual computation for agent 1
        path1 = paths[0]
        pos_at_t = path1[min(t, len(path1) - 1)]
        optimal_set = set(astar(maze, pos_at_t, maze.exit_pos))
        other_cells = {c for i, p in enumerate(paths) if i != 0 for c in p}
        post_exit = path1[t + 1:]
        expected_redundant = len({c for c in post_exit if c in other_cells and c not in optimal_set})

        results = RedundantComputationReduction(maze).compute(self.make_run(agents))
        a1 = next(r for r in results if r["agent_id"] == "1")

        assert a1["redundant_cells"] == expected_redundant
        assert a1["t_first_exit"] == t
        assert a1["pos_at_first_exit"] == list(pos_at_t)

    def test_pre_exit_cells_excluded(self, maze):
        """Cells shared before T_first_exit must not inflate the redundancy count."""
        paths = [astar(maze, maze.start_positions[i], maze.exit_pos) for i in range(3)]
        agents = [make_agent_result(str(i + 1), paths[i]) for i in range(3)]
        t = min(a["steps"] for a in agents)  # 16

        results = RedundantComputationReduction(maze).compute(self.make_run(agents))
        a1 = next(r for r in results if r["agent_id"] == "1")

        # Build an upper bound that would include pre-exit overlaps (wrong behaviour)
        path1 = paths[0]
        pos_at_t = path1[min(t, len(path1) - 1)]
        optimal_set = set(astar(maze, pos_at_t, maze.exit_pos))
        other_cells = {c for i, p in enumerate(paths) if i != 0 for c in p}

        # Wrongly counting ALL (not just post-exit) overlapping non-optimal cells
        inflated = len({c for c in set(path1) if c in other_cells and c not in optimal_set})

        # The correct count must not exceed the inflated upper bound,
        # and if there are pre-exit overlaps it must be strictly less.
        pre_exit_overlap_not_optimal = {
            c for c in set(path1[: t + 1]) if c in other_cells and c not in optimal_set
        }
        if pre_exit_overlap_not_optimal:
            assert a1["redundant_cells"] < inflated
        else:
            assert a1["redundant_cells"] <= inflated