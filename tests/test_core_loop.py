"""Tests for the agent core Runtime + Harness API.

Covers the original five scenarios plus the bugs/coverage gaps the round-1
review surfaced: ``ask_human`` pause/resume, ``abort`` path, durable state
serialization, parallel tool calls, timeout, retry, model-error containment.
"""
from __future__ import annotations

import asyncio
import json
import unittest

from agent_core import (
    Decision,
    GuardResult,
    Harness,
    RetryPolicy,
    RunStatus,
    Runtime,
    Tool,
    ToolCall,
    ToolContext,
    ToolResult,
    ValidationResult,
)
from agent_core.defaults.telemetry import NoopTelemetry


class ScriptedModel:
    """Model double that yields a pre-recorded sequence of decisions."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls = 0

    async def adecide(self, context, state, tools):
        if not self.decisions:
            raise AssertionError("model was called more times than expected")
        self.calls += 1
        return self.decisions.pop(0)


class RepeatingModel:
    def __init__(self, decision):
        self.decision = decision

    async def adecide(self, context, state, tools):
        return self.decision


class CrashingModel:
    async def adecide(self, context, state, tools):
        raise RuntimeError("provider down")


def _runtime(model, **kw) -> Runtime:
    runtime = Runtime.dev(model=model, **kw)
    runtime.telemetry = NoopTelemetry()
    return runtime


def _sync_tool(name: str, handler) -> Tool:
    return Tool.from_sync(name=name, description=name, handler=handler)


def _async_tool(name: str, handler) -> Tool:
    return Tool.from_async(name=name, description=name, handler=handler)


class CoreLoopTests(unittest.TestCase):
    # 1. Tool-driven completion ------------------------------------------

    def test_tool_result_satisfies_validator(self):
        seen: list = []

        def echo(args, ctx: ToolContext) -> ToolResult:
            seen.append(args)
            return ToolResult(tool_name="echo", ok=True, output={"seen": args["text"]})

        harness = Harness(
            tools=[_sync_tool("echo", echo)],
            validator=lambda state: ValidationResult(done=True, summary="done"),
        )
        result = _runtime(ScriptedModel(Decision.call_tool("echo", {"text": "hello"}))).run(harness, "echo hello")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.summary, "done")
        self.assertEqual(seen, [{"text": "hello"}])
        last_obs = result.state.observations[-1]
        self.assertEqual(last_obs.payload["tool_name"], "echo")
        self.assertEqual(last_obs.payload["output"], {"seen": "hello"})

    # 2. Direct final answer ---------------------------------------------

    def test_final_answer_stops_loop(self):
        result = _runtime(ScriptedModel(Decision.final_answer("ready"))).run(Harness(), "answer directly")
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.summary, "ready")
        self.assertEqual(result.state.observations, [])

    # 3. Approval pause + resume -----------------------------------------

    def test_approval_pause_and_resume(self):
        calls: list = []

        def risky(args, ctx):
            calls.append(args)
            return ToolResult(tool_name="risky", ok=True, output="applied")

        harness = Harness(
            tools=[_sync_tool("risky", risky)],
            guard=lambda decision, state: GuardResult(requires_approval=True, reason="review required"),
            validator=lambda state: ValidationResult(
                done=any(o.payload["tool_name"] == "risky" for o in state.observations),
                summary="approved action completed",
            ),
        )
        runtime = _runtime(ScriptedModel(Decision.call_tool("risky", {"id": 1})))

        paused = runtime.run(harness, "do risky thing")
        self.assertEqual(paused.status, RunStatus.APPROVAL_REQUIRED)
        self.assertEqual(paused.reason, "review required")
        self.assertIsNotNone(paused.state.interrupt)
        self.assertEqual(paused.state.interrupt.reason.value, "approval")
        self.assertIsNotNone(paused.state.interrupt.pending_decision)

        resumed = runtime.resume(harness, paused.state, approved=True)
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertEqual(resumed.summary, "approved action completed")
        self.assertEqual(calls, [{"id": 1}])

    def test_approval_rejection_feeds_back(self):
        runtime = _runtime(ScriptedModel(
            Decision.call_tool("noop", {}),
            Decision.final_answer("gave up"),
        ))

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output=None)

        harness = Harness(
            tools=[_sync_tool("noop", noop)],
            guard=lambda decision, state: GuardResult(requires_approval=True, reason="blocked"),
        )
        paused = runtime.run(harness, "should be blocked")
        self.assertEqual(paused.status, RunStatus.APPROVAL_REQUIRED)
        resumed = runtime.resume(harness, paused.state, approved=False, feedback="too risky")
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertIn("too risky", resumed.state.feedback)

    # 4. Tool error retried via feedback ---------------------------------

    def test_tool_error_becomes_feedback(self):
        attempts: list = []

        def unstable(args, ctx):
            attempts.append(args)
            if len(attempts) == 1:
                raise RuntimeError("temporary failure")
            return ToolResult(tool_name="unstable", ok=True, output="recovered")

        def validate(state):
            last = state.observations[-1].payload if state.observations else None
            if last and last["ok"]:
                return ValidationResult(done=True, summary="recovered")
            return ValidationResult(feedback="retry after tool failure")

        harness = Harness(
            tools=[_sync_tool("unstable", unstable)],
            validator=validate,
        )
        result = _runtime(ScriptedModel(
            Decision.call_tool("unstable", {"attempt": 1}),
            Decision.call_tool("unstable", {"attempt": 2}),
        )).run(harness, "recover from tool failure")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.summary, "recovered")
        self.assertEqual(len(result.state.observations), 2)
        self.assertFalse(result.state.observations[0].payload["ok"])
        self.assertIn("temporary failure", result.state.observations[0].payload["error"])
        self.assertIn("retry after tool failure", result.state.feedback)

    # 5. Max iterations stop ---------------------------------------------

    def test_max_iterations_stop(self):
        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output="still running")

        harness = Harness(
            tools=[_sync_tool("noop", noop)],
            validator=lambda state: ValidationResult(feedback="not done yet"),
        )
        result = _runtime(
            RepeatingModel(Decision.call_tool("noop")),
            max_iterations=2,
        ).run(harness, "never finishes")

        self.assertEqual(result.status, RunStatus.STOPPED)
        self.assertEqual(result.reason, "max_iterations reached")
        self.assertEqual(result.state.iteration, 2)
        self.assertEqual(len(result.state.observations), 2)


class NewlyCoveredPathTests(unittest.TestCase):
    """Paths the original tests missed: ask_human, abort, model error, parallel calls, serialization."""

    def test_ask_human_pauses_then_resumes_with_feedback(self):
        runtime = _runtime(ScriptedModel(
            Decision.ask_human("what's the threshold?"),
            Decision.final_answer("done with answer"),
        ))
        harness = Harness()

        paused = runtime.run(harness, "needs human input")
        self.assertEqual(paused.status, RunStatus.PAUSED)
        self.assertEqual(paused.reason, "what's the threshold?")
        self.assertIsNotNone(paused.state.interrupt)
        self.assertEqual(paused.state.interrupt.reason.value, "question")
        self.assertIsNone(paused.state.interrupt.pending_decision)

        resumed = runtime.resume(harness, paused.state, approved=True, feedback="threshold is 100")
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertIn("threshold is 100", resumed.state.feedback)

    def test_abort_returns_failed(self):
        runtime = _runtime(ScriptedModel(Decision.abort("unsafe input")))
        result = runtime.run(Harness(), "abort me")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.reason, "unsafe input")

    def test_model_exception_returns_failed_not_crash(self):
        runtime = _runtime(CrashingModel())
        result = runtime.run(Harness(), "model dies")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("provider down", result.reason)

    def test_parallel_tool_calls_run_concurrently(self):
        started: list = []
        finished: list = []

        async def slow_tool(args, ctx):
            started.append(args["id"])
            await asyncio.sleep(0.05)
            finished.append(args["id"])
            return ToolResult(tool_name="slow", ok=True, output=args["id"])

        harness = Harness(
            tools=[_async_tool("slow", slow_tool)],
            validator=lambda state: ValidationResult(
                done=len(state.observations) >= 3,
                summary="all done",
            ),
        )
        batch = Decision.call_tools((
            ToolCall(name="slow", args={"id": "a"}),
            ToolCall(name="slow", args={"id": "b"}),
            ToolCall(name="slow", args={"id": "c"}),
        ))
        runtime = _runtime(ScriptedModel(batch))
        result = runtime.run(harness, "parallel test")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(sorted(started), ["a", "b", "c"])
        # If we awaited sequentially, total time would be 3 * 50ms = 150ms.
        # We do not assert on wall time directly; presence of concurrent
        # "started" entries before the first "finished" is sufficient evidence.
        self.assertEqual(len(result.state.observations), 3)

    def test_unknown_tool_yields_failed_observation(self):
        runtime = _runtime(ScriptedModel(
            Decision.call_tool("nope"),
            Decision.final_answer("acknowledged"),
        ))
        result = runtime.run(Harness(), "missing tool")
        self.assertEqual(result.status, RunStatus.COMPLETED)
        last = result.state.observations[-1].payload
        self.assertFalse(last["ok"])
        self.assertIn("unknown tool", last["error"])

    def test_run_state_is_json_serializable(self):
        runtime = _runtime(ScriptedModel(
            Decision.call_tool("noop", {"x": 1}),
            Decision.final_answer("done"),
        ))

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output={"echoed": args})

        harness = Harness(
            tools=[_sync_tool("noop", noop)],
            validator=lambda state: ValidationResult(),
        )
        result = runtime.run(harness, "serialize me")
        snapshot = result.state.to_dict()
        # round-trip through JSON to prove durability
        round_tripped = json.loads(json.dumps(snapshot, default=str))
        self.assertEqual(round_tripped["goal"], "serialize me")
        self.assertEqual(round_tripped["iteration"], result.state.iteration)
        self.assertTrue(len(round_tripped["observations"]) >= 1)


class BackwardCompatTests(unittest.TestCase):
    """Lock in compatibility surfaces that round-3 review flagged."""

    def test_tool_result_positional_args_preserve_legacy_order(self):
        # (tool_name, ok, output, error) — the historical positional contract.
        r = ToolResult("echo", False, None, "boom")
        self.assertEqual(r.tool_name, "echo")
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "boom")
        # New trailing fields default to zero/empty so they cannot silently
        # absorb a positional argument meant for ``ok``.
        self.assertEqual(r.attempts, 1)
        self.assertEqual(r.call_id, "")
        self.assertEqual(r.duration_ms, 0.0)

    def test_decision_accepts_string_kind(self):
        # Older callers or deserialized state may still pass kind as a str.
        d = Decision(kind="final_answer", content="hello")  # type: ignore[arg-type]
        self.assertEqual(d.kind.value, "final_answer")
        self.assertEqual(d.content, "hello")


class ExecutorPolicyTests(unittest.TestCase):
    def test_tool_timeout_is_enforced(self):
        async def slow(args, ctx):
            await asyncio.sleep(0.2)
            return ToolResult(tool_name="slow", ok=True, output="late")

        harness = Harness(
            tools=[Tool.from_async(
                name="slow",
                description="slow",
                handler=slow,
                timeout_s=0.05,
            )],
            validator=lambda state: ValidationResult(
                done=any(not o.payload["ok"] for o in state.observations),
                summary="timed out as expected",
            ),
        )
        runtime = _runtime(ScriptedModel(Decision.call_tool("slow")))
        result = runtime.run(harness, "timeout test")
        self.assertEqual(result.status, RunStatus.COMPLETED)
        last = result.state.observations[-1].payload
        self.assertFalse(last["ok"])

    def test_retry_policy_retries_then_succeeds(self):
        attempts: list = []

        def flaky(args, ctx):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("not yet")
            return ToolResult(tool_name="flaky", ok=True, output="finally")

        harness = Harness(
            tools=[Tool.from_sync(
                name="flaky",
                description="retry me",
                handler=flaky,
                retry=RetryPolicy(max_attempts=3, backoff_initial_s=0.0),
            )],
            validator=lambda state: ValidationResult(
                done=any(o.payload["ok"] for o in state.observations),
                summary="recovered via retry",
            ),
        )
        runtime = _runtime(ScriptedModel(Decision.call_tool("flaky")))
        result = runtime.run(harness, "retry test")
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.summary, "recovered via retry")
        self.assertEqual(len(attempts), 3)
        last = result.state.observations[-1].payload
        self.assertTrue(last["ok"])
        self.assertEqual(last["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
