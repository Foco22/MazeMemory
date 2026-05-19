import asyncio
import litellm
from src.maze.generator import Maze

COLORS = {
    "1": "\033[92m",   # green
    "2": "\033[94m",   # blue
    "3": "\033[93m",   # yellow
}
RESET  = "\033[0m"
RED    = "\033[91m"
BOLD   = "\033[1m"


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = litellm.model_cost.get(model, {})
    return (
        prompt_tokens     * prices.get("input_cost_per_token", 0) +
        completion_tokens * prices.get("output_cost_per_token", 0)
    )


class LiveMazeView:
    AGENT_IDS = ["1", "2", "3"]

    def __init__(self, maze: Maze, model: str = ""):
        self.maze = maze
        self.model = model
        self.positions: dict[str, tuple[int, int]] = {}
        self.steps: dict[str, int] = {}
        self.prompt_tokens: dict[str, int] = {}
        self.completion_tokens: dict[str, int] = {}
        self.context_messages: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._rendered_lines = 0

    async def update(self, agent_id: str, position: tuple[int, int], timestep: int,
                     prompt_tokens: int = 0, completion_tokens: int = 0, context_messages: int = 0) -> None:
        async with self._lock:
            self.positions[agent_id] = position
            self.steps[agent_id] = timestep
            self.prompt_tokens[agent_id] = prompt_tokens
            self.completion_tokens[agent_id] = completion_tokens
            self.context_messages[agent_id] = context_messages
            self._render()

    def _render(self) -> None:
        if self._rendered_lines:
            print(f"\033[{self._rendered_lines}A", end="")

        pos_to_agent = {pos: aid for aid, pos in self.positions.items()}
        lines = 0

        for y in range(self.maze.rows):
            row = ""
            for x in range(self.maze.cols):
                pos = (x, y)
                if pos == self.maze.exit_pos:
                    row += f"{RED}E{RESET}"
                elif pos in pos_to_agent:
                    aid = pos_to_agent[pos]
                    row += f"{COLORS.get(aid, '')}{aid}{RESET}"
                elif self.maze.grid[y][x] == 1:
                    row += "█"
                else:
                    row += " "
            print(row)
            lines += 1

        print()
        lines += 1

        total_prompt = 0
        total_completion = 0
        for aid in self.AGENT_IDS:
            color = COLORS.get(aid, "")
            pos = self.positions.get(aid, "-")
            steps = self.steps.get(aid, 0)
            pt = self.prompt_tokens.get(aid, 0)
            ct = self.completion_tokens.get(aid, 0)
            msgs = self.context_messages.get(aid, 0)
            cost = _cost(self.model, pt, ct)
            total_prompt += pt
            total_completion += ct
            print(f"  {color}Agent {aid}{RESET}  pos={pos}  steps={steps}  ctx={msgs}msgs  prompt={pt}  completion={ct}  cost=${cost:.5f}")
            lines += 1

        print()
        lines += 1
        total_cost = _cost(self.model, total_prompt, total_completion)
        print(f"  {BOLD}TOTAL{RESET}  prompt={total_prompt}  completion={total_completion}  cost=${total_cost:.5f}")
        lines += 1

        self._rendered_lines = lines