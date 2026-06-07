import litellm
from datetime import datetime
from src.maze.generator import Maze
from src.maze.pathfinding import optimal_steps
from src.memory.shared import SharedMemoryStore
from src.agents.prompts import observer_system_prompt


class ObserverAgent:
    def __init__(self, maze: Maze, model: str, shared_memory: SharedMemoryStore, llm_kwargs: dict | None = None):
        self.maze = maze
        self.model = model
        self.shared_memory = shared_memory
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._llm_kwargs: dict = llm_kwargs or {}

    async def get_insight(self, requesting_agent_id: str, position: tuple[int, int]) -> str:
        trajectories = await self.shared_memory.get_all()

        statuses = self._compute_agent_statuses(trajectories)

        closest = min(
            (s for s in statuses if s["distance_to_exit"] is not None),
            key=lambda s: s["distance_to_exit"],
            default=None,
        )
        exit_found = closest and closest["distance_to_exit"] == 0

        visual_map = self._render_with_agents(trajectories, show_exit=exit_found)

        my_status = next((s for s in statuses if s["agent_id"] == requesting_agent_id), None)
        my_distance = my_status["distance_to_exit"] if my_status else None
        closest_teammate = min(
            (s for s in statuses if s["agent_id"] != requesting_agent_id and s["distance_to_exit"] is not None),
            key=lambda s: s["distance_to_exit"],
            default=None,
        )

        status_lines = []
        for s in statuses:
            tag = " ← CLOSEST TO EXIT" if closest and s["agent_id"] == closest["agent_id"] else ""
            status_lines.append(
                f"  Agent {s['agent_id']}: position={s['position']}, "
                f"distance_to_exit={s['distance_to_exit']}, steps_taken={s['steps_taken']}{tag}"
            )

        distance_note = (
            f"Agent {closest['agent_id']} is closest to the exit ({closest['distance_to_exit']} steps away)."
            if closest else "No agent has distance data yet."
        )

        if exit_found:
            successful_traj = trajectories.get(closest["agent_id"], [])
            waypoint = self._nearest_waypoint_on_path(position, successful_traj)
            exit_dir = self._relative_direction(position, self.maze.exit_pos)
            if waypoint and waypoint != self.maze.exit_pos:
                wp_dir = self._relative_direction(position, waypoint)
                coordination_context = (
                    f"PHASE 2 — EXIT FOUND: Agent {closest['agent_id']} has a confirmed navigable path to the exit.\n"
                    f"Step 1: Reach waypoint {waypoint} on that confirmed path — it is {wp_dir} of your current position.\n"
                    f"Step 2: From {waypoint}, follow the explored corridor (·) to the exit at {self.maze.exit_pos}.\n"
                    f"IMPORTANT: Do NOT navigate directly to the exit using cardinal direction — "
                    f"walls may block. Use the confirmed waypoint route."
                )
            else:
                coordination_context = (
                    f"PHASE 2 — EXIT FOUND at {self.maze.exit_pos} ({exit_dir}). Head there now."
                )
        elif closest_teammate is not None and (my_distance is None or closest_teammate["distance_to_exit"] < my_distance):
            waypoint = closest_teammate["position"]
            direction = self._relative_direction(position, waypoint)
            coordination_context = (
                f"PHASE 1 — EXPLORING: No agent has found the exit yet.\n"
                f"Agent {closest_teammate['agent_id']} is closest to exit "
                f"({closest_teammate['distance_to_exit']} steps) at {waypoint} — "
                f"that is {direction} of your position.\n"
                f"Head toward their region to exploit their progress."
            )
        else:
            coordination_context = (
                f"PHASE 1 — EXPLORING: You are the closest agent to the exit ({my_distance} steps). "
                f"You are the scout — keep exploring. Other agents are following your progress."
            )

        adj = self._adjacent_open_cells(position)

        user_message = (
            f"Agent {requesting_agent_id} is at {position} and requests navigation insight.\n\n"
            f"=== VISUAL MAP ===\n"
            f"Legend: █=wall  ·=explored  E=exit (Phase 2 only)  1/2/3=agent current position\n"
            f"NOTE: █ means wall OR unexplored — use ADJACENT OPEN CELLS below for reliable direction info.\n"
            f"{visual_map}\n\n"
            f"=== ADJACENT OPEN CELLS ===\n"
            f"{adj}\n\n"
            f"=== AGENT STATUS ===\n"
            + "\n".join(status_lines) + "\n\n"
            + f"=== DISTANCE ANALYSIS ===\n"
            + f"{distance_note}\n\n"
            + f"=== COORDINATION CONTEXT ===\n"
            + coordination_context + "\n\n"
            + f"Produce the 5-section report for Agent {requesting_agent_id}."
        )

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("trace.log", "a", encoding="utf-8") as f:
            f.write(f"\n[{ts}] [OBSERVER] Agent {requesting_agent_id} at {position}\n")
            f.write("--- USER MESSAGE ---\n")
            f.write(user_message + "\n")
            f.write("--- END USER MESSAGE ---\n")

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": observer_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            **self._llm_kwargs,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens

        recommendation = response.choices[0].message.content
        with open("trace.log", "a", encoding="utf-8") as f:
            f.write("--- OBSERVER RESPONSE ---\n")
            f.write(recommendation + "\n")
            f.write(f"\n[MAZE MAP]\n{visual_map}\n")
            f.write("--- END OBSERVER RESPONSE ---\n")
        return recommendation + f"\n\n[MAZE MAP]\n{visual_map}"

    def _relative_direction(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
        """Cardinal/intercardinal direction from from_pos to to_pos, no pathfinding."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        v = "south" if dy > 0 else ("north" if dy < 0 else "")
        h = "east" if dx > 0 else ("west" if dx < 0 else "")
        return f"{v}-{h}".strip("-") or "same position"

    def _render_with_agents(self, trajectories: dict, show_exit: bool = True) -> str:
        """Maze with x/y coordinate axes and explored cells (·) overlaid.

        show_exit=False hides the exit marker so agents don't learn its location before Phase 2.
        """
        agent_current: dict[str, tuple[int, int]] = {}
        all_visited: set[tuple[int, int]] = set()

        for agent_id, steps in trajectories.items():
            if steps:
                agent_current[agent_id] = (steps[-1][0], steps[-1][1])
                all_visited.update((x, y) for x, y, _ in steps)

        current_positions = {pos: aid for aid, pos in agent_current.items()}

        cols = self.maze.cols
        prefix = "    "  # space for the y-axis label (2 digits + 2 spaces)

        # x-axis header: two rows so 2-digit column numbers align correctly
        x_tens  = prefix + "".join(str(x // 10) if x >= 10 else " " for x in range(cols))
        x_units = prefix + "".join(str(x % 10) for x in range(cols))

        lines = [x_tens, x_units]
        for y in range(self.maze.rows):
            row = f"{y:2d}  "
            for x in range(cols):
                pos = (x, y)
                if pos == self.maze.exit_pos and show_exit:
                    row += "E"
                elif pos in current_positions:
                    row += current_positions[pos]
                elif pos in all_visited:
                    row += "·"
                else:
                    row += "█"
            lines.append(row)
        return "\n".join(lines)

    def _adjacent_open_cells(self, position: tuple[int, int]) -> str:
        """One line per direction: open/wall status from actual maze data (no inference needed)."""
        x, y = position
        directions = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        lines = []
        for name, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if self.maze.is_open(nx, ny):
                dist = optimal_steps(self.maze, (nx, ny), self.maze.exit_pos)
                dist_str = f"dist_to_exit={dist}" if dist is not None else "dist_to_exit=unknown"
                lines.append(f"  {name}: ({nx},{ny}) OPEN — {dist_str}")
            else:
                lines.append(f"  {name}: WALL")
        return "\n".join(lines)

    def _nearest_waypoint_on_path(
        self, position: tuple[int, int], trajectory: list[tuple[int, int, int]]
    ) -> tuple[int, int] | None:
        """Cell on trajectory nearest to position by Manhattan distance (no A* — fast)."""
        cells = [(x, y) for x, y, _ in trajectory]
        if not cells:
            return None
        return min(cells, key=lambda wp: abs(wp[0] - position[0]) + abs(wp[1] - position[1]))

    def _compute_agent_statuses(self, trajectories: dict) -> list[dict]:
        """Per-agent position, distance to exit, and step count from shared memory."""
        statuses = []
        for agent_id, steps in trajectories.items():
            if not steps:
                continue
            x, y, _ = steps[-1]
            pos = (x, y)
            dist = optimal_steps(self.maze, pos, self.maze.exit_pos)
            statuses.append({
                "agent_id": agent_id,
                "position": pos,
                "steps_taken": len(steps),
                "distance_to_exit": dist,
            })
        return sorted(statuses, key=lambda s: s["agent_id"])
