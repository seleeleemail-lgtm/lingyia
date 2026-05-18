"""Tests for streaming and tool concurrency limits."""
from __future__ import annotations

import asyncio
import time
import unittest

from lingyia_core import (
    Decision,
    Harness,
    Runtime,
    RunStatus,
    Tool,
    ToolCall,
    ToolResult,
    ToolUseBlock,
    ValidationResult,
)
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_core.executor import ToolExecutor


# Streaming --------------------------------------------------------------


class StreamingTests(unittest.TestCase):
    def test_astream_yields_events_then_final(self):
        class OneTurnModel:
            async def adecide(self, ctx, state, tools):
                return Decision.call_tools([
                    ToolUseBlock(id="call-1", name="noop", input={}),
                ])

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output="x")

        harness = Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop)],
            validator=lambda s: ValidationResult(done=True, summary="ok"),
        )
        rt = Runtime.dev(model=OneTurnModel(), max_iterations=3)
        rt.telemetry = NoopTelemetry()

        async def collect():
            events = []
            final = None
            async for kind, payload in rt.astream(harness, goal="t"):
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
