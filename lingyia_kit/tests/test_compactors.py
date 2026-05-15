"""Tests for TokenAwareCompactor."""
from __future__ import annotations

import unittest

from lingyia_core import Observation, RunState
from lingyia_kit.compactors import (
    TokenAwareCompactor,
    char_div4_estimator,
    tiktoken_estimator,
)


def _state_with_observations(n: int, payload_text: str = "x" * 1000) -> RunState:
    s = RunState(goal="some goal", run_id="r1")
    for i in range(n):
        s.observations.append(Observation(
            iteration=i,
            kind="tool_result",
            payload={"data": payload_text, "i": i},
        ))
    return s


class CharEstimatorTests(unittest.TestCase):
    def test_char_div4(self):
        self.assertEqual(char_div4_estimator(""), 1)  # min 1
        self.assertEqual(char_div4_estimator("abcd"), 1)
        self.assertEqual(char_div4_estimator("a" * 4000), 1000)


class TokenAwareCompactorTests(unittest.TestCase):
    def test_does_not_compact_when_under_budget(self):
        s = _state_with_observations(2, payload_text="short")
        compactor = TokenAwareCompactor(max_tokens=10_000)
        self.assertFalse(compactor.should_compact(s))

    def test_compacts_when_over_budget(self):
        # 50 observations * ~1k chars each * 1/4 ≈ 12500 tokens
        s = _state_with_observations(50, payload_text="x" * 1000)
        compactor = TokenAwareCompactor(max_tokens=5_000, keep_last_observations=5)
        self.assertTrue(compactor.should_compact(s))

        compacted = compactor.compact(s)
        # 5 kept + 1 marker = 6
        self.assertEqual(len(compacted.observations), 6)
        self.assertEqual(compacted.observations[0].kind, "compaction_marker")
        self.assertEqual(compacted.observations[0].payload["observations_dropped"], 45)
        # Original state should not have been mutated
        self.assertEqual(len(s.observations), 50)

    def test_feedback_also_truncated(self):
        s = RunState(goal="g")
        for i in range(30):
            s.feedback.append(f"fb-{i}")
        compactor = TokenAwareCompactor(
            max_tokens=1,  # force compaction
            keep_last_observations=0,
            keep_recent_feedback=5,
        )
        compacted = compactor.compact(s)
        self.assertEqual(len(compacted.feedback), 5)
        self.assertEqual(compacted.feedback[0], "fb-25")

    def test_custom_estimator_is_used(self):
        s = _state_with_observations(5, payload_text="x" * 100)
        # Estimator that says every string is 99999 tokens
        big = lambda _t: 99999
        compactor = TokenAwareCompactor(max_tokens=10_000, token_estimator=big)
        self.assertTrue(compactor.should_compact(s))

    def test_invalid_max_tokens_raises(self):
        with self.assertRaises(ValueError):
            TokenAwareCompactor(max_tokens=0)
        with self.assertRaises(ValueError):
            TokenAwareCompactor(max_tokens=-100)

    def test_runtime_invokes_compactor_when_threshold_hit(self):
        """Sanity: dropping a compactor into a Runtime makes it run."""
        import asyncio
        from lingyia_core import (
            Decision,
            Harness,
            Runtime,
            Tool,
            ToolResult,
            ValidationResult,
        )
        from lingyia_core.defaults.telemetry import NoopTelemetry

        # Force compaction every turn
        compactor = TokenAwareCompactor(
            max_tokens=10,
            keep_last_observations=2,
        )
        # Pre-seed observations into state via a manual harness setup
        calls = []

        class TwoTurnModel:
            def __init__(self):
                self.n = 0

            async def adecide(self, ctx, state, tools):
                calls.append(len(state.observations))
                self.n += 1
                if self.n >= 4:
                    return Decision.final_answer("done")
                return Decision.call_tool("noop", {})

        def noop_handler(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output="x" * 1000)

        rt = Runtime.dev(
            model=TwoTurnModel(),
            max_iterations=10,
        )
        rt.compactor = compactor
        rt.telemetry = NoopTelemetry()
        harness = Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop_handler)],
            validator=lambda s: ValidationResult(),
        )
        result = asyncio.run(rt.arun(harness, goal="t"))
        # After several iterations + compaction, observations bound by keep_last+1 marker
        self.assertLessEqual(len(result.state.observations), 3)


class TiktokenEstimatorTests(unittest.TestCase):
    def test_falls_back_when_tiktoken_unavailable(self):
        # Without tiktoken installed this returns the char fallback.
        est = tiktoken_estimator("gpt-4o")
        # Should at least be callable and return positive integer
        self.assertIsInstance(est("hello world"), int)
        self.assertGreater(est("hello world"), 0)


if __name__ == "__main__":
    unittest.main()
