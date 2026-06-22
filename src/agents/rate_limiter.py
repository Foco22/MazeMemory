import asyncio
import time


class AsyncRateLimiter:
    """Sliding-window rate limiter shared across concurrent async tasks."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._calls = [t for t in self._calls if now - t < self.period]
            if len(self._calls) >= self.max_calls:
                wait = self.period - (now - self._calls[0]) + 0.1
                await asyncio.sleep(wait)
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self.period]
            self._calls.append(now)