import pytest
from src.maze.generator import generate_maze
from src.maze.pathfinding import astar, optimal_steps
from src.maze.instances import get_maze, available_maze_ids


def test_maze_generation_is_deterministic():
    maze1 = generate_maze(seed=42)
    maze2 = generate_maze(seed=42)
    assert maze1.grid == maze2.grid


def test_different_seeds_produce_different_mazes():
    maze1 = generate_maze(seed=42)
    maze2 = generate_maze(seed=99)
    assert maze1.grid != maze2.grid


def test_exit_and_starts_are_open():
    maze = generate_maze(seed=42)
    assert maze.is_open(*maze.exit_pos)
    for pos in maze.start_positions:
        assert maze.is_open(*pos)


def test_exit_not_in_start_positions():
    maze = generate_maze(seed=42)
    assert maze.exit_pos not in maze.start_positions


def test_astar_finds_path():
    maze = generate_maze(seed=42)
    path = astar(maze, maze.start_positions[0], maze.exit_pos)
    assert len(path) > 0
    assert path[0] == maze.start_positions[0]
    assert path[-1] == maze.exit_pos


def test_astar_path_is_contiguous():
    maze = generate_maze(seed=42)
    path = astar(maze, maze.start_positions[0], maze.exit_pos)
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_optimal_steps_all_agents():
    maze = generate_maze(seed=42)
    for start in maze.start_positions:
        steps = optimal_steps(maze, start, maze.exit_pos)
        assert steps is not None and steps > 0


def test_all_maze_instances_load():
    for maze_id in available_maze_ids():
        maze = get_maze(maze_id)
        assert maze.grid is not None
        assert maze.exit_pos is not None