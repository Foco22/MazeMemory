from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.maze.generator import Maze

FPS = 4

_AGENT_COLORS = {"1": "#FFD700", "2": "#FFA500", "3": "#00CC44"}
_WALL_COLOR   = 0.15
_OPEN_COLOR   = 0.95
_EXIT_COLOR   = "#CC2222"
_START_ALPHA  = 0.35


def render_run_video(maze: Maze, result: dict, output_path: Path) -> None:
    paths = {ag["agent_id"]: ag["path"] for ag in result["agents"]}
    max_t = max(len(p) for p in paths.values())

    bg = np.full((maze.rows, maze.cols), _OPEN_COLOR)
    for y in range(maze.rows):
        for x in range(maze.cols):
            if maze.grid[y][x] == 1:
                bg[y][x] = _WALL_COLOR

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest")

    ex, ey = maze.exit_pos
    ax.add_patch(plt.Rectangle((ex - 0.5, ey - 0.5), 1, 1, color=_EXIT_COLOR, zorder=2))
    ax.text(ex, ey, "E", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=3)

    for i, sp in enumerate(maze.start_positions):
        sx, sy = sp
        aid = str(i + 1)
        ax.add_patch(plt.Rectangle(
            (sx - 0.5, sy - 0.5), 1, 1,
            color=_AGENT_COLORS[aid], alpha=_START_ALPHA, zorder=2,
        ))

    dots = {}
    for aid, color in _AGENT_COLORS.items():
        if aid in paths:
            x0, y0 = paths[aid][0]
            dot, = ax.plot(x0, y0, "o", color=color, markersize=9, zorder=4,
                           markeredgecolor="black", markeredgewidth=0.5)
            dots[aid] = dot

    scenario  = result.get("scenario", "")
    maze_id   = result.get("maze_id", "")
    run_num   = result.get("run_number", "")
    title_obj = ax.set_title(f"{scenario} | maze {maze_id} | run {run_num} | t=0", fontsize=9)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, maze.cols - 0.5)
    ax.set_ylim(maze.rows - 0.5, -0.5)

    def update(t):
        for aid, dot in dots.items():
            path = paths[aid]
            x, y = path[min(t, len(path) - 1)]
            dot.set_data([x], [y])
        title_obj.set_text(f"{scenario} | maze {maze_id} | run {run_num} | t={t}")
        return list(dots.values()) + [title_obj]

    ani = animation.FuncAnimation(fig, update, frames=max_t, interval=1000 // FPS, blit=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
