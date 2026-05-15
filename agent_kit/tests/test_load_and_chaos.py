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

from agent_core import (
    Decision,
    Harness,
    RetryPolicy,
    Runtime,
    RunStatus,
    Tool,
    ToolResult,
    ValidationResult,
)
from agent_core.defaults.telemetry import NoopTelemetry
from agent_kit.checkpointers import SqliteCheckpointer


# Load -------------------------------------------------------------------


class LoadTests(unittest.TestCase):
    def test_50_concurrent_agents_share_runtime(self):
        """50 agents share one Runtime + one SqliteCheckpointer. No
        race conditions, no state corruption, all complete."""

        class _Model:
            async def adecide(self, ctx, state, tools):
                return Decision.call_tool("noop", {"goal": state.goal})

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output=args.get("goal", ""))

        harness = Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop)],
            validator=lambda s: ValidationResult(
                done=bool(s.observations),
                summary=s.observations[-1].payload.get("output", "") if s.observations else "",
            ),
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
            goals = sorted(r.state.goal for r in results)
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
        harness = Harness(
            tools=[tool],
            validator=lambda s: ValidationResult(
                done=bool(s.observations) and s.observations[-1].payload["ok"],
                summary="recovered",
            ),
        )

        class _Model:
            async def adecide(self, ctx, state, tools):
                return Decision.call_tool("flaky", {})

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
