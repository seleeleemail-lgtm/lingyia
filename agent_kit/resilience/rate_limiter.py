"""Async token-bucket rate limiter.

Provider rate limits are usually expressed as requests per second / minute.
This limiter enforces a smooth target rate with a configurable burst budget.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional


class TokenBucketRateLimiter:
    def __init__(
        self,
        rate_per_sec: float,
        burst: float | None = None,
        *,
        clock=time.monotonic,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self.rate = float(rate_per_sec)
        self.burst = float(burst) if burst is not None else float(rate_per_sec)
        if self.burst <= 0:
            raise ValueError("burst must be > 0")
        self._tokens = self.burst
        self._clock = clock
        self._last = clock()
        # Lazy lock creation: ``asyncio.Lock()`` requires a running event loop
        # on Python 3.9. Defer instantiation to the first ``acquire()`` call
        # so the limiter is safe to construct in sync code (e.g. at module load).
        self._lock: Optional[asyncio.Lock] = None

    async def acquire(self, n: float = 1.0) -> None:
        """Block until ``n`` tokens are available, then consume them."""
        if n <= 0:
            return
        if n > self.burst:
            raise ValueError(f"n={n} exceeds burst={self.burst}")
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            while True:
                now = self._clock()
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                    self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                # Sleep just long enough for the deficit to refill.
                deficit = n - self._tokens
                wait_s = deficit / self.rate
                await asyncio.sleep(wait_s)

    @property
    def available_tokens(self) -> float:
        """Approximate token count (subject to lock contention)."""
        now = self._clock()
        elapsed = now - self._last
        return min(self.burst, self._tokens + max(0.0, elapsed) * self.rate)
