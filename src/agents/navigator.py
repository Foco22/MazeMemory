import os
import sys
import json
from datetime import datetime
import litellm
from src.maze.generator import Maze
from src.maze.pathfinding import optimal_steps
from src.memory.shared import SharedMemoryStore
from src.agents.tools import build_tools, build_shared_memory_tool, build_insight_tool
from src.agents.prompts import navigator_system_prompt
from src.agents.context import compress_messages, print_messages

MAX_STEPS = 500  # safety limit to prevent infinite loops

DIRECTIONS = {
    "north": (0, -1),
    "south": (0, 1),
    "east":  (1, 0),
    "west":  (-1, 0),
}


class NavigatorAgent:
    def __init__(
        self,
        agent_id: str,
        agent_index: int,
        maze: Maze,
        model: str,
        lookahead: int = 3,
        shared_memory: SharedMemoryStore | None = None,
        observer=None,
        on_move=None,
        llm_kwargs: dict | None = None,
    ):
        self.agent_id = agent_id
        self.maze = maze
        self.model = model
        self.lookahead = lookahead
        self._llm_kwargs: dict = llm_kwargs or {}
        self.shared_memory = shared_memory
        self.observer = observer
        self.on_move = on_move

        self.position = maze.start_positions[agent_index]
        self.path: list[tuple[int, int]] = [self.position]
        self.timestep = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.trace: list[dict] = []
        self._action_step = 0
        self._context_messages = 0
        self._llm_turns = 0

    async def run(self) -> dict:
        messages = [
            {
                "role": "system",
                "content": navigator_system_prompt(
                    self.agent_id,
                    self.lookahead,
                    has_shared_memory=self.shared_memory is not None,
                    has_observer=self.observer is not None,
                ),
            },
            {"role": "user", "content": "Start navigating. Find the exit."},
        ]
        tools = build_tools(self.lookahead)
        if self.shared_memory:
            tools.append(build_shared_memory_tool())
        if self.observer:
            tools.append(build_insight_tool())

        agent_started_at = datetime.now()
        while self.position != self.maze.exit_pos and len(self.path) <= MAX_STEPS:
            compressed = compress_messages(messages)
            response = await litellm.acompletion(
                model=self.model,
                messages=compressed,
                tools=tools,
                **self._llm_kwargs,
            )
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            self._context_messages = len(messages)
            self._llm_turns += 1

            # debug: uncomment to see token counts and compressed messages
            # actual = response.usage.prompt_tokens
            # print(f"[A{self.agent_id}] turn={self._llm_turns}  msgs_full={len(messages)}  msgs_compressed={len(compressed)}  tokens={actual}")
            # print_messages(compressed, label=f"A{self.agent_id} turn {self._llm_turns}")

            # debug: uncomment to dump full message history at turn 10
            # if self._llm_turns == 10:
            #     with open("trace.log", "a") as f:
            #         f.write(f"\n{'='*60}\n")
            #         f.write(f"[A{self.agent_id}] FULL MESSAGES AT TURN 10 ({len(messages)} msgs)\n")
            #         f.write(f"{'='*60}\n")
            #         for i, msg in enumerate(messages):
            #             f.write(f"  [{i}] {msg}\n")
            #         f.write(f"{'='*60}\n")

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                break

            llm_text = message.content or None
            turn_prompt     = response.usage.prompt_tokens
            turn_completion = response.usage.completion_tokens
            tool_results = []
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                result = await self._execute(tool_call)
                self.trace.append({
                    "step":             self._action_step,
                    "llm_text":         llm_text,
                    "tool_name":        tool_call.function.name,
                    "tool_args":        args,
                    "tool_result":      result,
                    "prompt_tokens":    turn_prompt,
                    "completion_tokens": turn_completion,
                })
                self._action_step += 1
                llm_text        = None   # only on first tool call per LLM turn
                turn_prompt     = None   # only on first tool call per LLM turn
                turn_completion = None
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
            messages.extend(tool_results)

        duration = (datetime.now() - agent_started_at).total_seconds()

        return {
            "agent_id": self.agent_id,
            "path": self.path,
            "steps": len(self.path) - 1,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "reached_exit": self.position == self.maze.exit_pos,
            "duration_seconds": round(duration, 3),
            "trace": self.trace,
        }

    async def _execute(self, tool_call) -> dict:
        raw_name = tool_call.function.name or ""
        # normalize malformed names like "move(north)\n</parameter>"
        import re as _re
        clean = _re.split(r'[\(\n<]', raw_name)[0].strip()
        name = clean if clean else raw_name

        args = json.loads(tool_call.function.arguments or "{}")
        # if direction was embedded in the name (e.g. "move(north)"), extract it
        if name == "move" and "direction" not in args:
            m = _re.search(r'\((north|south|east|west)\)', raw_name, _re.IGNORECASE)
            if m:
                args["direction"] = m.group(1).lower()

        if name == "get_location":
            result = {"x": self.position[0], "y": self.position[1]}
        elif name == "get_distance_to_exit":
            result = {"steps": optimal_steps(self.maze, self.position, self.maze.exit_pos)}
        elif name == "get_surroundings":
            result = self._surroundings()
        elif name == "move":
            result = await self._move(args["direction"])
        elif name == "get_shared_memory" and self.shared_memory:
            all_data = await self.shared_memory.get_all()
            result = {
                f"agent_{aid}": {
                    "current": {"x": positions[-1][0], "y": positions[-1][1]},
                    "visited": [{"x": x, "y": y} for x, y, _ in positions],
                }
                for aid, positions in all_data.items()
                if aid != self.agent_id and positions
            }
        elif name == "get_insight" and self.observer:
            recommendation = await self.observer.get_insight(self.agent_id, self.position)
            result = {"recommendation": recommendation}
        else:
            result = {"error": f"unknown tool: {name}"}

        _trace_filter = os.environ.get("TRACE_AGENT", "")
        if not _trace_filter or _trace_filter == self.agent_id:
            with open("trace.log", "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{ts}] [A{self.agent_id}] {name}({args}) → {result}", file=f, flush=True)
        return result

    def _surroundings(self) -> dict:
        x, y = self.position
        result = {}
        for direction, (dx, dy) in DIRECTIONS.items():
            count = 0
            for i in range(1, self.lookahead + 1):
                if self.maze.is_open(x + dx * i, y + dy * i):
                    count += 1
                else:
                    break
            result[direction] = count
        return result

    async def _move(self, direction: str) -> dict:
        dx, dy = DIRECTIONS[direction]
        x, y = self.position
        nx, ny = x + dx, y + dy

        if self.maze.is_open(nx, ny):
            self.position = (nx, ny)
            self.timestep += 1
            self.path.append(self.position)
            if self.shared_memory:
                await self.shared_memory.record(self.agent_id, nx, ny, self.timestep)
            if self.on_move:
                await self.on_move(self.agent_id, self.position, self.timestep, self.prompt_tokens, self.completion_tokens, self._context_messages)
            return {"success": True, "position": {"x": nx, "y": ny}}

        return {"success": False, "position": {"x": x, "y": y}, "reason": "wall"}