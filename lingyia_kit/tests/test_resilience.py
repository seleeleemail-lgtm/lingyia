"""Tests for circuit breaker, rate limiter, and ProtectedModel."""
from __future__ import annotations

import asyncio
import time
import unittest

from lingyia_core import Decision, RunState
from lingyia_kit.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ProtectedModel,
    TokenBucketRateLimiter,
)


# Mock clock helper
class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# Circuit breaker --------------------------------------------------------


class CircuitBreakerTests(unittest.TestCase):
    def test_starts_closed(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow())

    def test_opens_after_threshold_failures(self):
        clock = FakeClock()
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_s=10, clock=clock)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.allow())

    def test_success_in_closed_resets_failure_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_transitions_to_half_open_after_timeout(self):
        clock = FakeClock()
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=5, clock=clock)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        clock.advance(6)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

    def test_failure_in_half_open_reopens(self):
        clock = FakeClock()
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=5, clock=clock)
        cb.record_failure()
        cb.record_failure()
        clock.advance(6)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_sufficient_successes_in_half_open_close_circuit(self):
        clock = FakeClock()
        cb = CircuitBreaker(
            failure_threshold=2,
            reset_timeout_s=5,
            success_threshold_in_half_open=2,
            clock=clock,
        )
        cb.record_failure()
        cb.record_failure()
        clock.advance(6)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_success()
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)


# Rate limiter -----------------------------------------------------------


class RateLimiterTests(unittest.TestCase):
    def test_acquire_one_when_available(self):
        async def go():
            rl = TokenBucketRateLimiter(rate_per_sec=10, burst=5)
            for _ in range(5):
                await rl.acquire()

        asyncio.run(asyncio.wait_for(go(), timeout=1.0))

    def test_blocks_when_empty_then_refills(self):
        async def go():
            rl = TokenBucketRateLimiter(rate_per_sec=100, burst=2)
            await rl.acquire()
            await rl.acquire()
            # Now bucket empty. Should wait for ~10ms then succeed.
            start = time.monotonic()
            await rl.acquire()
            return time.monotonic() - start

        elapsed = asyncio.run(asyncio.wait_for(go(), timeout=2.0))
        self.assertGreaterEqual(elapsed, 0.005)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            TokenBucketRateLimiter(rate_per_sec=0)
        with self.assertRaises(ValueError):
            TokenBucketRateLimiter(rate_per_sec=1, burst=0)

    def test_n_larger_than_burst_raises(self):
        async def go():
            rl = TokenBucketRateLimiter(rate_per_sec=10, burst=5)
            await rl.acquire(n=100)

        with self.assertRaises(ValueError):
            asyncio.run(go())


# ProtectedModel ---------------------------------------------------------


class _FlakyModel:
    def __init__(self):
        self.calls = 0

    async def adecide(self, context, state, tools):
        self.calls += 1
        if self.calls <= 3:
            raise RuntimeError(f"flaky-{self.calls}")
        return Decision.final_answer("ok")


class _AlwaysFailModel:
    async def adecide(self, *args, **kwargs):
        raise RuntimeError("always fail")


class _OKModel:
    async def adecide(self, *args, **kwargs):
        return Decision.final_answer("ok")


class ProtectedModelTests(unittest.TestCase):
    def test_records_failure_and_eventually_opens(self):
        clock = FakeClock()
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=clock)
        pm = ProtectedModel(_AlwaysFailModel(), circuit_breaker=cb)

        async def go():
            with self.assertRaises(RuntimeError):
                await pm.adecide({}, RunState(goal="t"), [])
            with self.assertRaises(RuntimeError):
                await pm.adecide({}, RunState(goal="t"), [])
            # Now circuit open
            with self.assertRaises(CircuitOpenError):
                await pm.adecide({}, RunState(goal="t"), [])

        asyncio.run(go())

    def test_rate_limiter_throttles_calls(self):
        rl = TokenBucketRateLimiter(rate_per_sec=20, burst=2)
        pm = ProtectedModel(_OKModel(), rate_limiter=rl)

        async def go():
            start = time.monotonic()
            for _ in range(4):
                await pm.adecide({}, RunState(goal="t"), [])
            return time.monotonic() - start

        elapsed = asyncio.run(asyncio.wait_for(go(), timeout=2.0))
        # 4 calls, burst=2, rate=20/sec -> need to wait 2 more refills ~100ms
        self.assertGreaterEqual(elapsed, 0.08)

    def test_success_after_failures_records_success(self):
        clock = FakeClock()
        cb = CircuitBreaker(failure_threshold=5, reset_timeout_s=10, clock=clock)
        flaky = _FlakyModel()
        pm = ProtectedModel(flaky, circuit_breaker=cb)

        async def go():
            for _ in range(3):
                try:
                    await pm.adecide({}, RunState(goal="t"), [])
                except RuntimeError:
                    pass
            result = await pm.adecide({}, RunState(goal="t"), [])
            return result

        result = asyncio.run(go())
        self.assertEqual(result.content, "ok")
        # Circuit still closed (didn't hit threshold of 5)
        self.assertEqual(cb.state, CircuitState.CLOSED)


if __name__ == "__main__":
    unittest.main()
