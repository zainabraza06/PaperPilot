"""Cooperative client-side rate limiting.

Each of the three providers publishes a different politeness policy, and
exceeding it gets an IP throttled or blocked rather than merely slowed.
``AsyncRateLimiter`` enforces a minimum interval between requests for a
single source; one instance is shared by every caller of that source.
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Allow at most one request per ``min_interval`` seconds.

    A monotonic clock plus a lock is enough here: request volume is low
    (a handful per search) and a token bucket would add burst behaviour we
    explicitly do not want against these APIs.
    """

    def __init__(self, min_interval: float) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    @classmethod
    def from_rate(cls, requests_per_second: float) -> AsyncRateLimiter:
        """Build a limiter from a requests-per-second allowance."""
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        return cls(1.0 / requests_per_second)

    @property
    def min_interval(self) -> float:
        return self._min_interval

    async def acquire(self) -> None:
        """Block until the caller is allowed to issue its request."""
        if self._min_interval == 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed_at - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            self._next_allowed_at = now + self._min_interval

    async def __aenter__(self) -> AsyncRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None
