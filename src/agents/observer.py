import litellm
from datetime import datetime
from src.maze.generator import Maze
from src.maze.pathfinding import optimal_steps, astar
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

        visual_map = self._render_with_agents(trajectories)
        statuses = self._compute_agent_statuses(trajectories)

        closest = min(
            (s for s in statuses if s["distance_to_exit"] is not None),
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
            f"Agent {closest['agent_id']} is closest to the exit "
            f"({closest['distance_to_exit']} steps away from {closest['position']})."
            if closest else "No agent has distance data yet."
        )

        exit_found = closest and closest["distance_to_exit"] == 0
        if exit_found:
            priority = (
                f"IMPORTANT: Agent {closest['agent_id']} has already reached the exit at {self.maze.exit_pos}. "
                f"Agent {requesting_agent_id} must now head directly to {self.maze.exit_pos}. "
                f"In section 4, give only one instruction: move toward the exit."
            )
        else:
            exit_x, exit_y = self.maze.exit_pos
            ax, ay = position
            h_dir = "east" if exit_x > ax else ("west" if exit_x < ax else None)
            v_dir = "south" if exit_y > ay else ("north" if exit_y < ay else None)
            dirs = [d for d in [v_dir, h_dir] if d]
            priority = (
                f"The exit is at {self.maze.exit_pos}. "
                f"From {position} the exit is {' and '.join(dirs) if dirs else 'here'} — "
                f"prioritise those directions."
            )

        checkpoints = self._compute_path_checkpoints(position)
        checkpoint_lines = "\n".join(
            f"  {i+1}. go to {cp}" for i, cp in enumerate(checkpoints)
        )

        user_message = (
            f"Agent {requesting_agent_id} is at {position} and requests navigation insight.\n\n"
            f"=== VISUAL MAP ===\n"
            f"Legend: █=wall  ·=explored  E=exit  1/2/3=agent current position\n"
            f"{visual_map}\n\n"
            f"=== AGENT STATUS ===\n"
            + "\n".join(status_lines) + "\n\n"
            + f"=== DISTANCE ANALYSIS ===\n"
            + f"{distance_note}\n\n"
            + f"=== PATH RECOMMENDATION ===\n"
            + f"Optimal route from {position} to exit {self.maze.exit_pos}:\n"
            + checkpoint_lines + "\n\n"
            + f"=== PRIORITY ===\n"
            + f"{priority}\n\n"
            + f"Produce the 4-section report for Agent {requesting_agent_id}. "
            + f"In section 4, instruct the agent to follow the PATH RECOMMENDATION above."
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

        return response.choices[0].message.content

    def _render_with_agents(self, trajectories: dict) -> str:
        """Maze with x/y coordinate axes and explored cells (·) overlaid."""
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
                if pos == self.maze.exit_pos:
                    row += "E"
                elif pos in current_positions:
                    row += current_positions[pos]
                elif pos in all_visited:
                    row += "·"
                else:
                    row += "█"  # walls and unexplored corridors both appear dark
            lines.append(row)
        return "\n".join(lines)

    def _compute_path_checkpoints(self, position: tuple[int, int], n: int = 3) -> list[tuple[int, int]]:
        """Return n evenly-spaced coordinates along the A* path to exit, ending at the exit."""
        path = astar(self.maze, position, self.maze.exit_pos)
        if len(path) < 2:
            return []
        step = max(1, len(path) // (n + 1))
        checkpoints = [path[step * i] for i in range(1, n + 1) if step * i < len(path)]
        if path[-1] not in checkpoints:
            checkpoints.append(path[-1])
        return checkpoints

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