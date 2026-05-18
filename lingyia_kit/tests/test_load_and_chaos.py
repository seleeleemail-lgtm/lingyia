"""Load and chaos tests for the agent runtime.

Load: 50 concurrent agents share a single Runtime+Checkpointer+Telemetry
without state corruption, missing observations, or crashes.

Chaos: tools randomly throw or time out; agent should retry / recover and
either complete or fail cleanly. No exception should escape arun.
"""
from __future__ import annotations

import asyncio
import random
import tempfile
import unittest
from pathlib import Path

from lingyia_core import (
    BlockKind,
    Decision,
    Harness,
    ModelCapabilities,
    RetryPolicy,
    Role,
    Runtime,
    RunStatus,
    TextBlock,
    Tool,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ValidationResult,
)
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_kit.checkpointers import SqliteCheckpointer


_FAKE_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
)


def _first_user_text(state) -> str:
    """Extract the first user-role TextBlock content as a string."""
    for msg in state.messages:
        if msg.role == Role.USER:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    return block.text
    return ""


def _last_tool_result(state):
    """Return the most recent ToolResultBlock in state.messages, or None."""
    for msg in reversed(state.messages):
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                return block
    return None


# Load -------------------------------------------------------------------


class LoadTests(unittest.TestCase):
    def test_50_concurrent_agents_share_runtime(self):
        """50 agents share one Runtime + one SqliteCheckpointer. No
        race conditions, no state corruption, all complete."""

        class _Model:
            capabilities = _FAKE_CAPS

            async def adecide(self, ctx, state, tools):
                return Decision.call_tools([
                    ToolUseBlock(
                        id="noop-1",
                        name="noop",
                        input={"goal": _first_user_text(state)},
                    ),
                ])

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output=args.get("goal", ""))

        def _validator(s):
            last = _last_tool_result(s)
            if last is None:
                return ValidationResult()
            return ValidationResult(done=True, summary=last.content)

        harness = Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop)],
            validator=_validator,
        )

        with tempfile.TemporaryDirectory() as d:
            ckpath = Path(d) / "load.db"
            ck = SqliteCheckpointer(ckpath)
            rt = Runtime.production(
                model=_Model(),
                checkpointer=ck,
                telemetry=NoopTelemetry(),
                max_iterations=3,
            )

            async def run_one(i):
                result = await rt.arun(harness, goal=f"task-{i}")
                await ck.asave(result.state.run_id, result.state)
                return result

            async def go():
                return await asyncio.gather(*[run_one(i) for i in range(50)])

            results = asyncio.run(asyncio.wait_for(go(), timeout=15.0))
            ck.close()

            self.assertEqual(len(results), 50)
            statuses = [r.status for r in results]
            self.assertTrue(all(s == RunStatus.COMPLETED for s in statuses))
            goals = sorted(_first_user_text(r.state) for r in results)
            self.assertEqual(goals, sorted(f"task-{i}" for i in range(50)))


# Chaos ------------------------------------------------------------------


class ChaosTests(unittest.TestCase):
    def test_random_tool_failures_retry_and_recover(self):
        rng = random.Random(42)
        attempts = {"n": 0}

        def flaky_tool(args, ctx):
            attempts["n"] += 1
            if rng.random() < 0.3 and attempts["n"] <= 5:
                raise RuntimeError(f"chaos {attempts['n']}")
            return ToolResult(tool_name="flaky", ok=True, output="ok")

        tool = Tool.from_sync(
            name="flaky",
            description="random failures",
            handler=flaky_tool,
            retry=RetryPolicy(max_attempts=4, backoff_initial_s=0.0),
        )

        def _chaos_validator(s):
            last = _last_tool_result(s)
            if last is None or last.is_error:
                return ValidationResult()
            return ValidationResult(done=True, summary="recovered")

        harness = Harness(
            tools=[tool],
            validator=_chaos_validator,
        )

        class _Model:
            capabilities = _FAKE_CAPS

            async def adecide(self, ctx, state, tools):
                return Decision.call_tools([
                    ToolUseBlock(id="flaky-1", name="flaky", input={}),
                ])

        rt = Runtime.dev(model=_Model(), max_iterations=5)
        rt.telemetry = NoopTelemetry()

        # Run 20 chaotic agents; all should either complete or fail cleanly,
        # but no exception should escape.
        async def go():
            return await asyncio.gather(
                *[rt.arun(harness, f"chaos-{i}") for i in range(20)],
                return_exceptions=True,
            )

        results = asyncio.run(asyncio.wait_for(go(), timeout=10.0))
        for r in results:
            # No bare exception escaped
            self.assertNotIsInstance(r, Exception)
        # Statuses are valid enum values
        for r in results:
            self.assertIn(r.status, {
                RunStatus.COMPLETED,
                RunStatus.STOPPED,
                RunStatus.FAILED,
            })

    def test_model_provider_intermittent_failure_returns_failed_not_crash(self):
        """If the model raises on a turn, the run returns FAILED, the loop
        does not crash."""

        class _BadModel:
            capabilities = _FAKE_CAPS

            def __init__(self):
                self.n = 0

            async def adecide(self, ctx, state, tools):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("provider 503")
                return Decision.final_answer("recovered")

        rt = Runtime.dev(model=_BadModel())
        rt.telemetry = NoopTelemetry()
        harness = Harness(validator=lambda s: ValidationResult())
        result = asyncio.run(rt.arun(harness, "t"))
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("provider 503", result.reason)


if __name__ == "__main__":
    unittest.main()
