"""Tests for streaming and tool concurrency limits."""
from __future__ import annotations

import asyncio
import time
import unittest

from lingyia_core import (
    BlockKind,
    Decision,
    Harness,
    ModelCapabilities,
    Runtime,
    RunStatus,
    Tool,
    ToolCall,
    ToolResult,
    ToolUseBlock,
    ValidationResult,
)


_FAKE_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
    emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
)
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_core.executor import ToolExecutor


# Streaming --------------------------------------------------------------


class StreamingTests(unittest.TestCase):
    def test_astream_yields_events_then_final(self):
        # v0.2-α: validator gates only at FINAL_ANSWER. The model emits a
        # tool call on turn 1 and FINAL_ANSWER on turn 2; the validator's
        # done=True is honored at the FINAL_ANSWER gate.
        class TwoTurnModel:
            capabilities = _FAKE_CAPS

            def __init__(self):
                self.n = 0

            async def adecide(self, ctx, state, tools):
                self.n += 1
                if self.n == 1:
                    return Decision.call_tools([
                        ToolUseBlock(id="call-1", name="noop", input={}),
                    ])
                return Decision.final_answer("ok")

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output="x")

        harness = Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop)],
            validator=lambda s: ValidationResult(done=True, summary="ok"),
        )
        rt = Runtime.dev(model=TwoTurnModel(), max_iterations=3)
        rt.telemetry = NoopTelemetry()

        async def collect():
            events = []
            final = None
            async for kind, payload in rt.astream(harness, "t"):
                if kind == "event":
                    events.append(payload)
                elif kind == "final":
                    final = payload
                    break
            return events, final

        events, final = asyncio.run(collect())
        self.assertIsNotNone(final)
        self.assertEqual(final.status, RunStatus.COMPLETED)
        kinds = [e.kind for e in events]
        # Should include at least one decision + tool_started + tool_completed
        self.assertIn("decision", kinds)
        self.assertIn("tool_completed", kinds)


# Tool concurrency -------------------------------------------------------


class ToolConcurrencyTests(unittest.TestCase):
    def test_max_concurrency_serializes_parallel_calls(self):
        active = 0
        peak = 0
        lock = asyncio.Lock() if False else None  # noqa: simplification

        async def slow_tool(args, ctx):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            return ToolResult(tool_name="slow", ok=True, output=args["i"])

        tool = Tool.from_async(name="slow", description="slow", handler=slow_tool)

        # Force a parallel batch of 6 calls, but cap at 2 concurrent.
        executor = ToolExecutor(default_timeout_s=10, max_concurrency=2)
        from lingyia_core import ToolContext

        async def go():
            ctx = ToolContext(run_id="r", iteration=0, messages=(), metadata={})
            return await asyncio.gather(*[
                executor.execute(tool, ToolCall(name="slow", args={"i": i}), ctx)
                for i in range(6)
            ])

        results = asyncio.run(asyncio.wait_for(go(), timeout=2.0))
        self.assertEqual(len(results), 6)
        self.assertTrue(all(r.ok for r in results))
        self.assertLessEqual(peak, 2, f"peak concurrency was {peak}, expected <= 2")

    def test_no_limit_when_max_concurrency_zero(self):
        active = 0
        peak = 0

        async def fast_tool(args, ctx):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return ToolResult(tool_name="fast", ok=True)

        tool = Tool.from_async(name="fast", description="f", handler=fast_tool)
        executor = ToolExecutor(default_timeout_s=5, max_concurrency=0)
        from lingyia_core import ToolContext

        async def go():
            ctx = ToolContext(run_id="r", iteration=0, messages=(), metadata={})
            return await asyncio.gather(*[
                executor.execute(tool, ToolCall(name="fast", args={}), ctx)
                for _ in range(5)
            ])

        asyncio.run(asyncio.wait_for(go(), timeout=2.0))
        # No cap: peak should be 5 (all in flight)
        self.assertEqual(peak, 5)


if __name__ == "__main__":
    unittest.main()
