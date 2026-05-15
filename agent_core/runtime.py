"""Runtime: domain-agnostic orchestrator for the agent loop.

Runtime owns ``how to run``: model turn, tool execution, guard checks,
validator, interrupts, checkpoint calls, compaction, telemetry, retries.

Harness owns ``what to run``: tools, validator, guard, context builder.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .executor import RetryPolicy, Tool, ToolExecutor
from .protocols import Checkpointer, Compactor, Model, TelemetrySink
from .state import (
    Decision,
    DecisionKind,
    Interrupt,
    InterruptReason,
    Observation,
    RunResult,
    RunState,
    RunStatus,
    TelemetryEvent,
    ToolCall,
    ToolContext,
    ToolResult,
)


GuardFn = Callable[[Decision, RunState], "GuardResult"]
ValidatorFn = Callable[[RunState], "ValidationResult"]
ContextBuilderFn = Callable[[RunState, Sequence[Tool]], Mapping[str, Any]]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool = True
    requires_approval: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ValidationResult:
    done: bool = False
    summary: str = ""
    feedback: str = ""
    needs_human: bool = False
    question: str = ""


def _default_context_builder(state: RunState, tools: Sequence[Tool]) -> Mapping[str, Any]:
    return {
        "goal": state.goal,
        "iteration": state.iteration,
        "observations": [obs.payload for obs in state.observations],
        "feedback": list(state.feedback),
        "tools": [
            {"name": t.name, "description": t.description, "input_schema": dict(t.input_schema)}
            for t in tools
        ],
    }


def _default_guard(decision: Decision, state: RunState) -> GuardResult:
    return GuardResult()


def _default_validator(state: RunState) -> ValidationResult:
    return ValidationResult()


@dataclass
class Harness:
    """Domain-specific configuration: tools + validator + guard + context builder.

    Harness deliberately knows nothing about timeouts, retries, checkpointers,
    telemetry, or compaction. Those live in the Runtime.

    ``granted_permissions`` gates tool execution. The wildcard ``"*"`` allows
    every tool (default). In production, list explicit permission strings and
    decline broad grants — Tool.required_permissions is checked against this
    set before any tool runs.
    """

    tools: list[Tool] = field(default_factory=list)
    validator: ValidatorFn = _default_validator
    guard: GuardFn = _default_guard
    context_builder: ContextBuilderFn = _default_context_builder
    granted_permissions: frozenset = field(default_factory=lambda: frozenset(["*"]))

    def tool_by_name(self, name: str) -> Optional[Tool]:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def permission_check(self, tool: Tool) -> Optional[str]:
        """Return None if allowed, or an error message describing the gap."""
        if "*" in self.granted_permissions:
            return None
        missing = tool.required_permissions - self.granted_permissions
        if missing:
            return f"missing permissions: {sorted(missing)}"
        return None


@dataclass
class Runtime:
    """Domain-agnostic orchestrator. Construct via ``Runtime.dev`` or ``Runtime.production``."""

    model: Model
    checkpointer: Checkpointer
    telemetry: TelemetrySink
    compactor: Compactor
    executor: ToolExecutor
    max_iterations: int = 8
    # 0 means no budget. When set, the run aborts as soon as accumulated
    # ``state.metadata["cost_usd"]`` exceeds this value.
    max_cost_usd: float = 0.0
    # Optional callback ``(ModelUsage) -> float`` that adapters can use to
    # compute cost when the adapter didn't fill ``usage.cost_usd`` itself.
    cost_estimator: Optional[Callable[[Any], float]] = None

    # Construction --------------------------------------------------------

    @classmethod
    def dev(
        cls,
        model: Model,
        max_iterations: int = 8,
        default_timeout_s: float = 60.0,
        default_retry: Optional[RetryPolicy] = None,
        max_cost_usd: float = 0.0,
        cost_estimator: Optional[Callable[[Any], float]] = None,
    ) -> "Runtime":
        """Dev/test runtime with permissive defaults.

        Uses in-memory checkpointing, stdout telemetry, and no compaction.
        Not safe for production.
        """
        from .defaults.checkpointer import InMemoryCheckpointer
        from .defaults.compactor import NoOpCompactor
        from .defaults.telemetry import StdoutTelemetry

        return cls(
            model=model,
            checkpointer=InMemoryCheckpointer(),
            telemetry=StdoutTelemetry(),
            compactor=NoOpCompactor(),
            executor=ToolExecutor(
                default_timeout_s=default_timeout_s,
                default_retry=default_retry,
            ),
            max_iterations=max_iterations,
            max_cost_usd=max_cost_usd,
            cost_estimator=cost_estimator,
        )

    @classmethod
    def production(
        cls,
        model: Model,
        checkpointer: Checkpointer,
        telemetry: TelemetrySink,
        compactor: Optional[Compactor] = None,
        default_timeout_s: float = 60.0,
        default_retry: Optional[RetryPolicy] = None,
        max_iterations: int = 8,
        max_cost_usd: float = 0.0,
        cost_estimator: Optional[Callable[[Any], float]] = None,
    ) -> "Runtime":
        """Production runtime: fail closed on missing durability/observability."""
        if checkpointer is None:
            raise ValueError("production runtime requires an explicit checkpointer")
        if telemetry is None:
            raise ValueError("production runtime requires an explicit telemetry sink")
        from .defaults.compactor import NoOpCompactor

        return cls(
            model=model,
            checkpointer=checkpointer,
            telemetry=telemetry,
            compactor=compactor or NoOpCompactor(),
            executor=ToolExecutor(
                default_timeout_s=default_timeout_s,
                default_retry=default_retry,
            ),
            max_iterations=max_iterations,
            max_cost_usd=max_cost_usd,
            cost_estimator=cost_estimator,
        )

    # Public API ----------------------------------------------------------

    def run(self, harness: Harness, goal: str) -> RunResult:
        """Synchronous entry point. Internally drives the async loop."""
        return asyncio.run(self.arun(harness, goal))

    async def arun(self, harness: Harness, goal: str) -> RunResult:
        state = RunState(goal=goal)
        return await self._continue(harness, state)

    def resume(
        self,
        harness: Harness,
        state: RunState,
        approved: bool = True,
        feedback: str = "",
    ) -> RunResult:
        return asyncio.run(self.aresume(harness, state, approved, feedback))

    async def aresume(
        self,
        harness: Harness,
        state: RunState,
        approved: bool = True,
        feedback: str = "",
    ) -> RunResult:
        interrupt = state.interrupt
        state.interrupt = None

        if interrupt is None:
            return await self._continue(harness, state)

        if interrupt.reason == InterruptReason.APPROVAL:
            if not approved:
                state.add_feedback(feedback or "approval rejected")
                state.iteration += 1
                return await self._continue(harness, state)
            if interrupt.pending_decision is None:
                state.add_feedback("approval interrupt missing pending decision")
                state.iteration += 1
                return await self._continue(harness, state)
            await self._execute_decision(harness, state, interrupt.pending_decision)
            finished = await self._maybe_finish_after_tools(harness, state)
            if finished is not None:
                return finished
            return await self._continue(harness, state)

        # QUESTION resume
        if feedback:
            state.add_feedback(feedback)
        state.iteration += 1
        return await self._continue(harness, state)

    # Loop ----------------------------------------------------------------

    async def _continue(self, harness: Harness, state: RunState) -> RunResult:
        while state.iteration < self.max_iterations:
            if self.compactor.should_compact(state):
                state = self.compactor.compact(state)
                self._emit(state, TelemetryEvent(kind="compacted", iteration=state.iteration))

            context = harness.context_builder(state, harness.tools)

            try:
                decision = await self.model.adecide(context, state, harness.tools)
            except asyncio.CancelledError:
                # Let cancellation propagate so callers can shut the loop down.
                raise
            except Exception as exc:
                self._emit(state, TelemetryEvent(
                    kind="model_error",
                    iteration=state.iteration,
                    payload={"error": str(exc)},
                ))
                return RunResult(
                    status=RunStatus.FAILED,
                    state=state,
                    reason=f"model error: {exc}",
                )

            usage = decision.usage
            cost_delta = 0.0
            if usage is not None:
                cost_delta = float(usage.cost_usd or 0.0)
                if cost_delta == 0.0 and self.cost_estimator is not None:
                    try:
                        cost_delta = float(self.cost_estimator(usage) or 0.0)
                    except Exception:
                        cost_delta = 0.0
                if cost_delta:
                    prior = float(state.metadata.get("cost_usd", 0.0) or 0.0)
                    state.metadata["cost_usd"] = round(prior + cost_delta, 6)
                # Always accumulate token counters so callers can report on
                # usage even when pricing is unknown.
                tokens = state.metadata.setdefault("tokens", {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                })
                tokens["prompt"] += int(usage.prompt_tokens or 0)
                tokens["completion"] += int(usage.completion_tokens or 0)
                tokens["cached"] += int(usage.cached_tokens or 0)

            decision_payload: dict[str, Any] = {
                "kind": decision.kind.value,
                "tool_calls": len(decision.tool_calls),
            }
            if usage is not None:
                decision_payload.update({
                    "model_id": usage.model_id,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "cost_usd": cost_delta,
                    "cost_total_usd": state.metadata.get("cost_usd", 0.0),
                })
            self._emit(state, TelemetryEvent(
                kind="decision",
                iteration=state.iteration,
                payload=decision_payload,
            ))

            # Budget enforcement: abort the run if accumulated cost exceeds
            # the configured ceiling. Done *after* recording the decision so
            # the run's trace shows the cost that triggered the abort.
            if self.max_cost_usd > 0:
                total = float(state.metadata.get("cost_usd", 0.0) or 0.0)
                if total > self.max_cost_usd:
                    return RunResult(
                        status=RunStatus.FAILED,
                        state=state,
                        reason=f"budget exceeded: ${total:.6f} > ${self.max_cost_usd:.6f}",
                    )

            if decision.kind == DecisionKind.FINAL_ANSWER:
                verdict = harness.validator(state)
                if verdict.needs_human:
                    state.interrupt = Interrupt(
                        reason=InterruptReason.QUESTION,
                        message=verdict.question,
                        iteration=state.iteration,
                    )
                    return RunResult(
                        status=RunStatus.PAUSED,
                        state=state,
                        reason=verdict.question,
                    )
                return RunResult(
                    status=RunStatus.COMPLETED,
                    state=state,
                    summary=verdict.summary or decision.content,
                )

            if decision.kind == DecisionKind.ASK_HUMAN:
                state.interrupt = Interrupt(
                    reason=InterruptReason.QUESTION,
                    message=decision.content,
                    iteration=state.iteration,
                )
                return RunResult(
                    status=RunStatus.PAUSED,
                    state=state,
                    reason=decision.content,
                )

            if decision.kind == DecisionKind.ABORT:
                return RunResult(
                    status=RunStatus.FAILED,
                    state=state,
                    reason=decision.content,
                )

            if decision.kind != DecisionKind.CALL_TOOL:
                return RunResult(
                    status=RunStatus.FAILED,
                    state=state,
                    reason=f"unknown decision kind: {decision.kind}",
                )

            guard = harness.guard(decision, state)
            if guard.requires_approval:
                state.interrupt = Interrupt(
                    reason=InterruptReason.APPROVAL,
                    message=guard.reason,
                    pending_decision=decision,
                    iteration=state.iteration,
                )
                return RunResult(
                    status=RunStatus.APPROVAL_REQUIRED,
                    state=state,
                    reason=guard.reason,
                )
            if not guard.allowed:
                state.add_feedback(guard.reason)
                self._emit(state, TelemetryEvent(
                    kind="guard_denied",
                    iteration=state.iteration,
                    payload={"reason": guard.reason},
                ))
                state.iteration += 1
                continue

            await self._execute_decision(harness, state, decision)
            finished = await self._maybe_finish_after_tools(harness, state)
            if finished is not None:
                return finished

        return RunResult(
            status=RunStatus.STOPPED,
            state=state,
            reason="max_iterations reached",
        )

    async def _execute_decision(
        self,
        harness: Harness,
        state: RunState,
        decision: Decision,
    ) -> None:
        """Execute all tool calls in a Decision in parallel and record observations."""
        if not decision.tool_calls:
            return
        ctx = ToolContext(
            run_id=state.run_id,
            iteration=state.iteration,
            goal=state.goal,
            metadata=dict(state.metadata),
        )
        tasks: list[Any] = []
        # Preserve call->result alignment so we can attach tool_args to each
        # observation. The model adapter needs tool_args to rebuild the
        # assistant turn when replaying conversation history.
        ordered_calls = list(decision.tool_calls)
        for call in ordered_calls:
            tool = harness.tool_by_name(call.name)
            if tool is None:
                tasks.append(_unknown_tool_result(call))
                continue
            perm_error = harness.permission_check(tool)
            if perm_error is not None:
                self._emit(state, TelemetryEvent(
                    kind="permission_denied",
                    iteration=state.iteration,
                    payload={"tool": tool.name, "reason": perm_error},
                ))
                tasks.append(_permission_denied_result(call, perm_error))
                continue
            tasks.append(
                self.executor.execute(
                    tool,
                    call,
                    ctx,
                    on_event=lambda e: self._emit(state, e),
                )
            )
        results = await asyncio.gather(*tasks)
        for call, result in zip(ordered_calls, results):
            state.add_observation(Observation(
                iteration=state.iteration,
                kind="tool_result",
                payload={
                    "tool_name": result.tool_name,
                    "tool_args": dict(call.args),
                    "call_id": result.call_id,
                    "ok": result.ok,
                    "output": result.output,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                    "attempts": result.attempts,
                },
            ))

    async def _maybe_finish_after_tools(
        self,
        harness: Harness,
        state: RunState,
    ) -> Optional[RunResult]:
        verdict = harness.validator(state)
        if verdict.done:
            return RunResult(
                status=RunStatus.COMPLETED,
                state=state,
                summary=verdict.summary,
            )
        if verdict.needs_human:
            state.interrupt = Interrupt(
                reason=InterruptReason.QUESTION,
                message=verdict.question,
                iteration=state.iteration,
            )
            return RunResult(
                status=RunStatus.PAUSED,
                state=state,
                reason=verdict.question,
            )
        state.add_feedback(verdict.feedback)
        state.iteration += 1
        return None

    def _emit(self, state: RunState, event: TelemetryEvent) -> None:
        # Tag each event with the run id so downstream sinks can correlate.
        enriched = (
            event
            if event.run_id == state.run_id
            else TelemetryEvent(
                kind=event.kind,
                iteration=event.iteration,
                timestamp=event.timestamp,
                payload=dict(event.payload),
                run_id=state.run_id,
            )
        )
        state.trace.append(enriched)
        self.telemetry.emit(enriched)


async def _unknown_tool_result(call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_name=call.name,
        call_id=call.call_id,
        ok=False,
        error=f"unknown tool: {call.name}",
    )


async def _permission_denied_result(call: ToolCall, reason: str) -> ToolResult:
    return ToolResult(
        tool_name=call.name,
        call_id=call.call_id,
        ok=False,
        error=f"permission denied: {reason}",
    )
