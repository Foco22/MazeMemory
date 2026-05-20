import asyncio
import json
from pathlib import Path


class SharedMemoryStore:
    def __init__(self, log_path: Path | None = None):
        self._lock = asyncio.Lock()
        self._data: dict[str, list[tuple[int, int, int]]] = {}  # agent_id -> [(x, y, timestep)]
        self._log_path = log_path

    async def record(self, agent_id: str, x: int, y: int, timestep: int) -> None:
        async with self._lock:
            if agent_id not in self._data:
                self._data[agent_id] = []
            self._data[agent_id].append((x, y, timestep))
            if self._log_path:
                with self._log_path.open("a") as f:
                    f.write(json.dumps({"agent_id": agent_id, "x": x, "y": y, "t": timestep}) + "\n")

    async def get_all(self) -> dict[str, list[tuple[int, int, int]]]:
        async with self._lock:
            return {k: list(v) for k, v in self._data.items()}