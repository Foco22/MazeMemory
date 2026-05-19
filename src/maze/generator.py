import random
from dataclasses import dataclass


@dataclass
class Maze:
    grid: list[list[int]]  # 0 = open, 1 = wall
    start_positions: list[tuple[int, int]]  # one per agent, (x, y)
    exit_pos: tuple[int, int]
    seed: int
    rows: int
    cols: int

    def is_open(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] == 0

    def neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        return [
            (x + dx, y + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if self.is_open(x + dx, y + dy)
        ]

    def render(self) -> str:
        symbols = {0: " ", 1: "█"}
        lines = []
        for y, row in enumerate(self.grid):
            line = ""
            for x, cell in enumerate(row):
                pos = (x, y)
                if pos == self.exit_pos:
                    line += "E"
                elif pos in self.start_positions:
                    line += str(self.start_positions.index(pos) + 1)
                else:
                    line += symbols[cell]
            lines.append(line)
        return "\n".join(lines)


def generate_maze(seed: int, rows: int = 15, cols: int = 15) -> Maze:
    assert rows % 2 == 1 and cols % 2 == 1, "rows and cols must be odd"

    rng = random.Random(seed)
    grid = [[1] * cols for _ in range(rows)]

    def carve(x: int, y: int) -> None:
        grid[y][x] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 1:
                grid[y + dy // 2][x + dx // 2] = 0
                carve(nx, ny)

    carve(1, 1)

    # Fixed positions so agents always start spread across the maze
    start_positions = [
        (1, 1),           # top-left
        (1, rows - 2),    # bottom-left
        (cols - 2, 1),    # top-right
    ]
    exit_pos = (cols - 2, rows - 2)  # bottom-right

    return Maze(
        grid=grid,
        start_positions=start_positions,
        exit_pos=exit_pos,
        seed=seed,
        rows=rows,
        cols=cols,
    )
