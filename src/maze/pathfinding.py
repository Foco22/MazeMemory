import heapq
from .generator import Maze


def astar(maze: Maze, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set = [(h(start), start)]
    came_from: dict[tuple, tuple] = {}
    g: dict[tuple, int] = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for neighbor in maze.neighbors(*current):
            tentative_g = g[current] + 1
            if tentative_g < g.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + h(neighbor), neighbor))

    return []


def optimal_steps(maze: Maze, start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    path = astar(maze, start, goal)
    return len(path) - 1 if path else None
