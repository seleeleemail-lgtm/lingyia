"""Tests for token usage, pricing, and budget enforcement."""
from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from lingyia_core import (
    BlockKind,
    Decision,
    Harness,
    Message,
    ModelCapabilities,
    Role,
    Runtime,
    RunStatus,
    TextBlock,
    Tool,
    ToolResult,
    ToolUseBlock,
    ValidationResult,
)
from lingyia_core.state import ModelUsage


_FAKE_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
    emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
)
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_kit.adapters._openai_base import OpenAICompatibleModel
from lingyia_kit.pricing import estimate_cost, register_pricing


class PricingTableTests(unittest.TestCase):
    def test_known_model_cost(self):
        # gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
        c = estimate_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
        self.assertAlmostEqual(c, 0.15, places=4)

    def test_cached_tokens_discounted(self):
        # 1M prompt total, 500k cached -> 500k at full + 500k at 10%
        c = estimate_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=500_000)
        expected = (500_000 / 1e6) * 0.15 + (500_000 / 1e6) * 0.015
        self.assertAlmostEqual(c, expected, places=4)

    def test_unknown_model_returns_zero_not_raise(self):
        c = estimate_cost("unknown/model-xyz", prompt_tokens=1000, completion_tokens=1000)
        self.assertEqual(c, 0.0)

    def test_register_pricing_overrides(self):
        register_pricing("my-test-model", 1.0, 2.0)
        c = estimate_cost("my-test-model", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertAlmostEqual(c, 3.0, places=4)


class OpenAIUsageParsingTests(unittest.TestCase):
    def _client(self, handler) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_adapter_extracts_usage_from_response(self):
        def handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            })

        model = OpenAICompatibleModel(
            api_key="k",
            base_url="https://x/v1",
            model="gpt-4o-mini",
            client=self._client(handler),
        )
        from lingyia_core import RunState
        seed = [Message(role=Role.USER, content=(TextBlock(text="t"),))]
        decision = asyncio.run(model.adecide({}, RunState(messages=seed), []))
        self.assertIsNotNone(decision.usage)
        self.assertEqual(decision.usage.prompt_tokens, 100)
        self.assertEqual(decision.usage.completion_tokens, 50)
        self.assertEqual(decision.usage.cached_tokens, 20)
        self.assertEqual(decision.usage.model_id, "gpt-4o-mini")


class CostAccumulationTests(unittest.TestCase):
    def _harness(self) -> Harness:
        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True)
        return Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop)],
            validator=lambda s: ValidationResult(),  # never done -> max_iter
        )

    def test_runtime_accumulates_cost_across_decisions(self):
        # Two decisions, each with 1M prompt tokens at gpt-4o-mini rate ($0.15)
        d1 = Decision(
            kind=Decision.final_answer("").kind,
            content=(TextBlock(text=""),),
            usage=ModelUsage(
                prompt_tokens=1_000_000,
                completion_tokens=0,
                model_id="gpt-4o-mini",
                cost_usd=0.15,
            ),
        )

        class OneShotModel:
            capabilities = _FAKE_CAPS

            async def adecide(self, ctx, state, tools):
                return d1

        rt = Runtime.dev(model=OneShotModel(), max_iterations=2)
        rt.telemetry = NoopTelemetry()
        result = asyncio.run(rt.arun(self._harness(), "t"))
        # final_answer ends loop on iter 0; only one decision was made
        self.assertAlmostEqual(result.state.metadata["cost_usd"], 0.15, places=6)
        self.assertEqual(result.state.metadata["tokens"]["prompt"], 1_000_000)

    def test_max_cost_usd_aborts_run(self):
        expensive = Decision.call_tools([
            ToolUseBlock(id="call-1", name="noop", input={}),
        ])
        expensive = Decision(
            kind=expensive.kind,
            content=expensive.content,
            usage=ModelUsage(
                prompt_tokens=2_000_000,
                completion_tokens=0,
                model_id="gpt-4o-mini",
                cost_usd=0.30,
            ),
        )

        class RepeatingModel:
            capabilities = _FAKE_CAPS

            async def adecide(self, ctx, state, tools):
                return expensive

        rt = Runtime.dev(model=RepeatingModel(), max_iterations=10, max_cost_usd=0.20)
        rt.telemetry = NoopTelemetry()
        result = asyncio.run(rt.arun(self._harness(), "t"))
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIn("budget exceeded", result.reason)
        self.assertGreater(result.state.metadata["cost_usd"], 0.20)

    def test_cost_estimator_used_when_decision_lacks_cost(self):
        decision = Decision(
            kind=Decision.final_answer("hi").kind,
            content=(TextBlock(text="hi"),),
            usage=ModelUsage(
                prompt_tokens=1_000_000,
                completion_tokens=0,
                model_id="gpt-4o-mini",
                # cost_usd=0 -> runtime will call estimator
            ),
        )

        class OneShotModel:
            capabilities = _FAKE_CAPS

            async def adecide(self, ctx, state, tools):
                return decision

        def estimator(usage):
            return estimate_cost(
                usage.model_id,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_tokens,
            )

        rt = Runtime.dev(
            model=OneShotModel(),
            max_iterations=2,
            cost_estimator=estimator,
        )
        rt.telemetry = NoopTelemetry()
        result = asyncio.run(rt.arun(self._harness(), "t"))
        self.assertAlmostEqual(result.state.metadata["cost_usd"], 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
