"""Tool definition + ToolExecutor.

Tools carry their own schema, timeout, retry policy, and side-effect metadata.
ToolExecutor handles timeout, retry, and sync/async adaptation around handler calls.

Note on sync timeouts: ``asyncio.wait_for`` cancels the awaiter, not the underlying
thread. If a sync tool times out, the wrapped function may still be running. The
executor emits a ``tool_timeout`` event with ``may_still_be_running=True`` so callers
can compensate if the tool has side effects.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

from .state import TelemetryEvent, ToolCall, ToolContext, ToolResult


SyncHandler = Callable[[Mapping[str, Any], ToolContext], ToolResult]
AsyncHandler = Callable[[Mapping[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    backoff_initial_s: float = 0.5
    backoff_multiplier: float = 2.0
    retryable_on: tuple[type[BaseException], ...] = ()

    def is_retryable(self, exc: BaseException) -> bool:
        if not self.retryable_on:
            return True
        return isinstance(exc, self.retryable_on)


@dataclass(frozen=True)
class Tool:
    """A tool the runtime can invoke on behalf of the model.

    Either ``handler`` runs as async or sync, determined by ``is_async``.
    Prefer ``Tool.from_async`` / ``Tool.from_sync`` factories.

    ``required_permissions`` is checked by the Runtime against the Harness's
    ``granted_permissions`` before each call. A Harness with the wildcard
    ``"*"`` grants everything (the default). Use specific permission strings
    (e.g. ``"fs.write"``, ``"network.write"``) in production.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    handler: Any = None  # AsyncHandler | SyncHandler
    is_async: bool = True
    timeout_s: Optional[float] = None
    retry: Optional[RetryPolicy] = None
    idempotent: bool = False
    side_effect: str = "read_only"  # "read_only" | "write" | "external_io"
    required_permissions: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_async(
        cls,
        name: str,
        description: str,
        handler: AsyncHandler,
        input_schema: Optional[Mapping[str, Any]] = None,
        timeout_s: Optional[float] = None,
        retry: Optional[RetryPolicy] = None,
        idempotent: bool = False,
        side_effect: str = "read_only",
        required_permissions: Optional[frozenset] = None,
    ) -> "Tool":
        return cls(
            name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
            handler=handler,
            is_async=True,
            timeout_s=timeout_s,
            retry=retry,
            idempotent=idempotent,
            side_effect=side_effect,
            required_permissions=frozenset(required_permissions or ()),
        )

    @classmethod
    def from_sync(
        cls,
        name: str,
        description: str,
        handler: SyncHandler,
        input_schema: Optional[Mapping[str, Any]] = None,
        timeout_s: Optional[float] = None,
        retry: Optional[RetryPolicy] = None,
        idempotent: bool = False,
        side_effect: str = "read_only",
        required_permissions: Optional[frozenset] = None,
    ) -> "Tool":
        return cls(
            name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
            handler=handler,
            is_async=False,
            timeout_s=timeout_s,
            retry=retry,
            idempotent=idempotent,
            side_effect=side_effect,
            required_permissions=frozenset(required_permissions or ()),
        )


EventCallback = Callable[[TelemetryEvent], None]


class ToolExecutor:
    """Executes a ToolCall with timeout + retry + sync/async adaptation.

    ``max_concurrency`` caps the number of tool invocations in flight at
    any moment across this executor. Set to ``0`` (default) to disable.
    Useful when many parallel tool_calls would overwhelm a downstream
    service (e.g. SaaS APIs with low concurrent-connection limits).
    """

    def __init__(
        self,
        default_timeout_s: float = 60.0,
        default_retry: Optional[RetryPolicy] = None,
        max_concurrency: int = 0,
    ) -> None:
        self.default_timeout_s = default_timeout_s
        self.default_retry = default_retry or RetryPolicy(max_attempts=1)
        self.max_concurrency = max_concurrency
        # Lazy: ``asyncio.Semaphore()`` needs a running event loop on 3.9.
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute(
        self,
        tool: Tool,
        call: ToolCall,
        ctx: ToolContext,
        on_event: Optional[EventCallback] = None,
    ) -> ToolResult:
        if self.max_concurrency > 0:
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self.max_concurrency)
            async with self._semaphore:
                return await self._execute_inner(tool, call, ctx, on_event)
        return await self._execute_inner(tool, call, ctx, on_event)

    async def _execute_inner(
        self,
        tool: Tool,
        call: ToolCall,
        ctx: ToolContext,
        on_event: Optional[EventCallback] = None,
    ) -> ToolResult:
        retry = tool.retry or self.default_retry
        timeout = tool.timeout_s or self.default_timeout_s
        start = time.perf_counter()
        last_error: Optional[BaseException] = None

        for attempt in range(1, retry.max_attempts + 1):
            if on_event:
                on_event(TelemetryEvent(
                    kind="tool_started",
                    iteration=ctx.iteration,
                    payload={
                        "tool": tool.name,
                        "call_id": call.call_id,
                        "attempt": attempt,
                        "timeout_s": timeout,
                    },
                ))
            try:
                result = await self._invoke(tool, call, ctx, timeout)
                duration_ms = (time.perf_counter() - start) * 1000.0
                final = ToolResult(
                    tool_name=tool.name,
                    call_id=call.call_id,
                    ok=result.ok,
                    output=result.output,
                    error=result.error,
                    duration_ms=duration_ms,
                    attempts=attempt,
                )
                if on_event:
                    on_event(TelemetryEvent(
                        kind="tool_completed",
                        iteration=ctx.iteration,
                        payload={
                            "tool": tool.name,
                            "call_id": call.call_id,
                            "ok": final.ok,
                            "duration_ms": duration_ms,
                            "attempts": attempt,
                        },
                    ))
                return final
            except asyncio.CancelledError:
                # Cancellation must propagate; do not absorb into a ToolResult
                # and never retry a cancelled task.
                raise
            except asyncio.TimeoutError as exc:
                last_error = exc
                if on_event:
                    on_event(TelemetryEvent(
                        kind="tool_timeout",
                        iteration=ctx.iteration,
                        payload={
                            "tool": tool.name,
                            "call_id": call.call_id,
                            "attempt": attempt,
                            "timeout_s": timeout,
                            "may_still_be_running": not tool.is_async,
                        },
                    ))
                if attempt >= retry.max_attempts or not retry.is_retryable(exc):
                    break
                await self._backoff(retry, attempt)
            except Exception as exc:
                last_error = exc
                if on_event:
                    on_event(TelemetryEvent(
                        kind="tool_error",
                        iteration=ctx.iteration,
                        payload={
                            "tool": tool.name,
                            "call_id": call.call_id,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    ))
                if attempt >= retry.max_attempts or not retry.is_retryable(exc):
                    break
                await self._backoff(retry, attempt)

        duration_ms = (time.perf_counter() - start) * 1000.0
        return ToolResult(
            tool_name=tool.name,
            call_id=call.call_id,
            ok=False,
            error=str(last_error) if last_error else "unknown error",
            duration_ms=duration_ms,
            attempts=retry.max_attempts,
        )

    async def _invoke(
        self,
        tool: Tool,
        call: ToolCall,
        ctx: ToolContext,
        timeout_s: float,
    ) -> ToolResult:
        args = dict(call.args)
        if tool.is_async:
            return await asyncio.wait_for(tool.handler(args, ctx), timeout=timeout_s)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, lambda: tool.handler(args, ctx))
        return await asyncio.wait_for(future, timeout=timeout_s)

    @staticmethod
    async def _backoff(policy: RetryPolicy, attempt: int) -> None:
        wait_s = policy.backoff_initial_s * (policy.backoff_multiplier ** (attempt - 1))
        if wait_s > 0:
            await asyncio.sleep(wait_s)
