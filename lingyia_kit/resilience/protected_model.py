"""Model wrapper that adds circuit-breaker + rate-limiter protection.

Drop-in for any object implementing the ``Model`` protocol. Use it when you
want production-grade fault isolation:

    real = SiliconFlowModel(api_key=...)
    cb = CircuitBreaker(failure_threshold=5, reset_timeout_s=30)
    rl = TokenBucketRateLimiter(rate_per_sec=10)
    runtime = Runtime.production(
        model=ProtectedModel(real, circuit_breaker=cb, rate_limiter=rl),
        ...,
    )
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from lingyia_core import Decision, RunState

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .rate_limiter import TokenBucketRateLimiter


class ProtectedModel:
    def __init__(
        self,
        model: Any,
        *,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
    ) -> None:
        self.model = model
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter

    async def adecide(
        self,
        context: Mapping[str, Any],
        state: RunState,
        tools: Sequence[Any],
    ) -> Decision:
        if self.circuit_breaker is not None and not self.circuit_breaker.allow():
            raise CircuitOpenError(
                f"circuit open (state={self.circuit_breaker.state.value})"
            )
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        try:
            result = await self.model.adecide(context, state, tools)
        except Exception:
            if self.circuit_breaker is not None:
                self.circuit_breaker.record_failure()
            raise
        else:
            if self.circuit_breaker is not None:
                self.circuit_breaker.record_success()
            return result
