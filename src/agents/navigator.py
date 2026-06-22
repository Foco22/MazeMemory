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
from src.agents.rate_limiter import AsyncRateLimiter

MAX_STEPS = 500  # safety limit to prevent infinite loops


def _log_context(agent_id: str, turn: int, messages: list, prompt_tokens: int) -> None:
    def _section(label):
        return f"  ··· {label} {'·'*(50 - len(label))}\n"

    with open("trace.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'═'*60}\n")
        f.write(f"[CONTEXT] A{agent_id}  turn={turn}  msgs={len(messages)}  tokens={prompt_tokens}\n")
        f.write(f"{'═'*60}\n")

        section = None
        for i, msg in enumerate(messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "?")
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            tool_calls = None if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else None
            text = str(content or "").replace("\n", " ")

            # detect section transitions
            new_section = None
            if i == 0:
                new_section = "SYSTEM PROMPT"
            elif i == 1:
                new_section = "COMPRESSED HISTORY"
            elif isinstance(content, str) and content.startswith("[LAST OBSERVER INSIGHT]"):
                new_section = "LAST OBSERVER INSIGHT"
            elif isinstance(content, str) and content.startswith("[CURRENT SHARED MEMORY]"):
                new_section = "CURRENT SHARED MEMORY"
            elif role == "assistant" and tool_calls and section in ("COMPRESSED HISTORY", "SYSTEM PROMPT", "START"):
                new_section = "RECENT FULL TURNS"
            elif role == "assistant" and tool_calls and section == "CURRENT SHARED MEMORY":
                new_section = "RECENT FULL TURNS"

            if new_section and new_section != section:
                section = new_section
                f.write(_section(section))

            if tool_calls:
                calls = ", ".join(f"{(tc['function']['name'] if isinstance(tc, dict) else tc.function.name)}({(tc['function']['arguments'] if isinstance(tc, dict) else tc.function.arguments)})" for tc in tool_calls)
                f.write(f"  [{i:02d}] assistant  → {calls}\n")
            elif tool_call_id:
                f.write(f"  [{i:02d}] tool result  {text}\n")
            else:
                f.write(f"  [{i:02d}] {role:10s}  {text}\n")

        f.write(f"{'═'*60}\n")

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
        raw_memory_tool: bool = True,
        on_move=None,
        on_llm_call=None,
        llm_kwargs: dict | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
    ):
        self.agent_id = agent_id
        self.maze = maze
        self.model = model
        self.lookahead = lookahead
        self._llm_kwargs: dict = llm_kwargs or {}
        self._rate_limiter = rate_limiter
        self.shared_memory = shared_memory
        self.observer = observer
        self.raw_memory_tool = raw_memory_tool
        self.on_move = on_move
        self.on_llm_call = on_llm_call

        self.position = maze.start_positions[agent_index]
        self.path: list[tuple[int, int]] = [self.position]
        self.timestep = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.trace: list[dict] = []
        self._action_step = 0
        self._context_messages = 0
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        self._llm_turns = 0

        # get_insight() cooldown: one call allowed every N moves, derived from maze size
        self._insight_interval = max(maze.rows, maze.cols) // 2
        self._last_insight_at = 0  # first call available only after _insight_interval moves

    async def run(self) -> dict:
        messages = [
            {
                "role": "system",
                "content": navigator_system_prompt(
                    self.agent_id,
                    self.lookahead,
                    has_shared_memory=self.shared_memory is not None,
                    has_observer=self.observer is not None,
                    insight_interval=self._insight_interval,
                ),
            },
            {"role": "user", "content": "Start navigating. Find the exit."},
        ]
        tools = build_tools(self.lookahead)
        if self.shared_memory and self.raw_memory_tool:
            tools.append(build_shared_memory_tool())
        if self.observer:
            tools.append(build_insight_tool())

        agent_started_at = datetime.now()
        while self.position != self.maze.exit_pos and len(self.path) <= MAX_STEPS:
            compressed = compress_messages(messages)
            if self._rate_limiter:
                await self._rate_limiter.acquire()
            response = await litellm.acompletion(
                model=self.model,
                messages=compressed,
                tools=tools,
                num_retries=6,
                timeout=120,
                **self._llm_kwargs,
            )
            self._last_prompt_tokens = response.usage.prompt_tokens
            self._last_completion_tokens = response.usage.completion_tokens
            self.prompt_tokens += self._last_prompt_tokens
            self.completion_tokens += self._last_completion_tokens
            _details = getattr(response.usage, "prompt_tokens_details", None)
            self.cache_hit_tokens += (getattr(_details, "cached_tokens", None) or 0) if _details else 0
            self.cache_miss_tokens += (getattr(_details, "cache_creation_tokens", None) or 0) if _details else 0
            self._context_messages = len(compressed)
            self._llm_turns += 1
            if self.on_llm_call:
                await self.on_llm_call(self.agent_id, self._last_prompt_tokens, self._last_completion_tokens, self._context_messages)

            if os.environ.get("TRACE_CONTEXT"):
                _log_context(self.agent_id, self._llm_turns, compressed, self._last_prompt_tokens)

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
            "cache_hit_tokens": self.cache_hit_tokens or None,
            "cache_miss_tokens": self.cache_miss_tokens or None,
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
            recent = self.path[-6:-1]  # last 5 cells before current position
            result = {
                "x": self.position[0],
                "y": self.position[1],
                "recent_path": [
                    {"x": x, "y": y, "dist_to_exit": optimal_steps(self.maze, (x, y), self.maze.exit_pos)}
                    for x, y in recent
                ],
            }
        elif name == "get_distance_to_exit":
            result = {"steps_to_exit": optimal_steps(self.maze, self.position, self.maze.exit_pos)}
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
            steps_since = self.timestep - self._last_insight_at
            if steps_since < self._insight_interval:
                remaining = self._insight_interval - steps_since
                result = {"cooldown": f"Not available yet. Move {remaining} more step(s) before calling get_insight again."}
            else:
                self._last_insight_at = self.timestep
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
                await self.on_move(self.agent_id, self.position, self.timestep,
                                   self.prompt_tokens, self.completion_tokens,
                                   self._context_messages,
                                   self._last_prompt_tokens, self._last_completion_tokens)
            recent = self.path[-6:-1]  # last 5 cells before current position
            return {
                "success": True,
                "position": {"x": nx, "y": ny},
                "recent_path": [
                    {"x": rx, "y": ry, "dist_to_exit": optimal_steps(self.maze, (rx, ry), self.maze.exit_pos)}
                    for rx, ry in recent
                ],
            }

        return {"success": False, "position": {"x": x, "y": y}, "reason": "wall"}