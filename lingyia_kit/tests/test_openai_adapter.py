"""Unit tests for the OpenAI-compatible adapter base.

Uses httpx MockTransport so no real network or API key is needed. Verifies:
- Tool schemas are constructed correctly from lingyia_core.Tool definitions.
- Tool-call responses parse into Decision.call_tools with right args.
- Plain text responses parse into Decision.final_answer.
- Multi-iteration conversation history is reconstructed in OpenAI shape
  from RunState.observations (including the tool_args we now persist).
"""
from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from lingyia_core import (
    Decision,
    DecisionKind,
    Observation,
    RunState,
    Tool,
    ToolContext,
    ToolResult,
)
from lingyia_kit.adapters._openai_base import OpenAICompatibleModel


def _client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def _make_model(handler) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        api_key="test",
        base_url="https://example.test/v1",
        model="test-model",
        client=_client(handler),
    )


def _noop_tool() -> Tool:
    def _h(args, ctx):  # pragma: no cover (never called in adapter tests)
        return ToolResult(tool_name="noop", ok=True)

    return Tool.from_sync(
        name="noop",
        description="No-op for schema tests.",
        handler=_h,
        input_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    )


class OpenAIAdapterTests(unittest.TestCase):
    def test_tool_call_response_becomes_decision_call_tools(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "noop",
                                "arguments": json.dumps({"x": 5}),
                            },
                        }],
                    },
                }],
            })

        model = _make_model(handler)
        state = RunState(goal="test")
        decision = asyncio.run(model.adecide({}, state, [_noop_tool()]))

        self.assertEqual(decision.kind, DecisionKind.CALL_TOOL)
        self.assertEqual(len(decision.tool_calls), 1)
        self.assertEqual(decision.tool_calls[0].name, "noop")
        self.assertEqual(decision.tool_calls[0].args, {"x": 5})
        self.assertEqual(decision.tool_calls[0].call_id, "call_1")
        # Auth header set
        self.assertEqual(captured["auth"], "Bearer test")
        # Tool schema in payload
        self.assertEqual(captured["payload"]["tools"][0]["function"]["name"], "noop")
        self.assertEqual(captured["payload"]["tool_choice"], "auto")

    def test_text_response_becomes_final_answer(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "all done"}}],
            })

        model = _make_model(handler)
        state = RunState(goal="test")
        decision = asyncio.run(model.adecide({}, state, []))
        self.assertEqual(decision.kind, DecisionKind.FINAL_ANSWER)
        self.assertEqual(decision.content, "all done")

    def test_history_replays_assistant_turn_and_tool_results(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            })

        # Seed a state that looks like iteration 0 has already executed a tool.
        state = RunState(goal="do thing")
        state.iteration = 1
        state.observations.append(Observation(
            iteration=0,
            kind="tool_result",
            payload={
                "tool_name": "noop",
                "tool_args": {"x": 42},
                "call_id": "call_zero",
                "ok": True,
                "output": "hello",
                "error": "",
            },
        ))

        model = _make_model(handler)
        asyncio.run(model.adecide({"system_prompt": "be helpful"}, state, [_noop_tool()]))

        msgs = captured["payload"]["messages"]
        # system, user, assistant(tool_calls), tool
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "be helpful")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "do thing")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["name"], "noop")
        self.assertEqual(
            json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]),
            {"x": 42},
        )
        self.assertEqual(msgs[3]["role"], "tool")
        self.assertEqual(msgs[3]["tool_call_id"], "call_zero")
        result_payload = json.loads(msgs[3]["content"])
        self.assertTrue(result_payload["ok"])
        self.assertEqual(result_payload["output"], "hello")

    def test_malformed_tool_arguments_degrade_gracefully(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_x",
                            "type": "function",
                            "function": {
                                "name": "noop",
                                "arguments": "not valid json",
                            },
                        }],
                    },
                }],
            })

        model = _make_model(handler)
        decision = asyncio.run(model.adecide({}, RunState(goal="t"), [_noop_tool()]))
        self.assertEqual(decision.kind, DecisionKind.CALL_TOOL)
        self.assertEqual(decision.tool_calls[0].args, {"_raw": "not valid json"})

    def test_provider_subclasses_set_base_url(self):
        from lingyia_kit.adapters import (
            MiniMaxModel,
            OpenAIModel,
            SiliconFlowModel,
        )

        sf = SiliconFlowModel(api_key="x")
        self.assertEqual(sf.base_url, "https://api.siliconflow.cn/v1")
        self.assertTrue(sf.model.startswith("Qwen/") or "/" in sf.model)

        mm = MiniMaxModel(api_key="x")
        self.assertEqual(mm.base_url, "https://api.minimaxi.com/v1")
        self.assertTrue(mm.model.startswith("MiniMax"))

        oa = OpenAIModel(api_key="x")
        self.assertEqual(oa.base_url, "https://api.openai.com/v1")
        self.assertTrue(oa.model.startswith("gpt"))


if __name__ == "__main__":
    unittest.main()
