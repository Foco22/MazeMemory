import asyncio
import atexit
import json
import sys
from pathlib import Path
from src.maze.generator import Maze

_PRICES_PATH = Path(__file__).parent.parent / "metrics" / "prices.json"
with _PRICES_PATH.open() as _f:
    _PRICES: dict[str, dict] = json.load(_f)

COLORS = {
    "1": "\033[92m",   # green
    "2": "\033[94m",   # blue
    "3": "\033[93m",   # yellow
}
RESET  = "\033[0m"
RED    = "\033[91m"
BOLD   = "\033[1m"


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = _PRICES.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def _enter_alt_screen() -> None:
    sys.stdout.write("\033[?1049h\033[H")
    sys.stdout.flush()


def _exit_alt_screen() -> None:
    sys.stdout.write("\033[?1049l")
    sys.stdout.flush()


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
        _enter_alt_screen()
        atexit.register(_exit_alt_screen)

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
        CLR = "\033[K"

        sys.stdout.write("\033[H\033[J")  # home + clear to end (safe inside alt screen)
        sys.stdout.flush()

        pos_to_agent = {pos: aid for aid, pos in self.positions.items()}

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
            print(row + CLR)

        print(CLR)

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
            print(f"  {color}Agent {aid}{RESET}  pos={pos}  steps={steps}  ctx={msgs}msgs  prompt={pt}  completion={ct}  cost=${cost:.5f}{CLR}")

        print(CLR)
        total_cost = _cost(self.model, total_prompt, total_completion)
        print(f"  {BOLD}TOTAL{RESET}  prompt={total_prompt}  completion={total_completion}  cost=${total_cost:.5f}{CLR}")