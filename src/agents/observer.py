import litellm
from src.maze.generator import Maze
from src.memory.shared import SharedMemoryStore
from src.agents.prompts import observer_system_prompt


class ObserverAgent:
    def __init__(self, maze: Maze, model: str, shared_memory: SharedMemoryStore):
        self.maze = maze
        self.model = model
        self.shared_memory = shared_memory
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def get_insight(self, requesting_agent_id: str, position: tuple[int, int]) -> str:
        trajectories = await self.shared_memory.get_all()

        trajectory_summary = "\n".join(
            f"  Agent {agent_id}: {len(steps)} steps recorded — "
            f"last position ({steps[-1][0]}, {steps[-1][1]})"
            for agent_id, steps in trajectories.items()
            if steps
        ) or "  No trajectory data available yet."

        visited_by_others = {
            (x, y)
            for agent_id, steps in trajectories.items()
            if agent_id != requesting_agent_id
            for x, y, _ in steps
        }

        user_message = (
            f"Agent {requesting_agent_id} is at position {position} and needs navigation advice.\n\n"
            f"Trajectory summary:\n{trajectory_summary}\n\n"
            f"Cells already explored by other agents: {len(visited_by_others)} cells.\n"
            f"Exit is at {self.maze.exit_pos}.\n\n"
            "Recommend a direction for this agent to move, prioritising unexplored areas."
        )

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": observer_system_prompt()},
                {"role": "user", "content": user_message},
            ],
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens

        return response.choices[0].message.content