"""End-to-end integration: full agent run with real-shaped components.

Exercises the production stack together:
- httpx MockTransport simulating OpenAI-compatible API
- SqliteCheckpointer for durability
- JsonlTelemetry for observability
- RegexRedactor for PII scrub
- TokenAwareCompactor for context bound
- Tool permission gating
- Cost accumulation

This is the closest thing to a smoke test for the whole framework.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from lingyia_core import (
    Decision,
    GuardResult,
    Harness,
    Role,
    Runtime,
    RunStatus,
    Tool,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ValidationResult,
)
from lingyia_kit.adapters._openai_base import OpenAICompatibleModel
from lingyia_kit.checkpointers import SqliteCheckpointer
from lingyia_kit.compactors import TokenAwareCompactor
from lingyia_kit.pricing import estimate_cost
from lingyia_kit.redaction import RegexRedactor
from lingyia_kit.telemetry import JsonlTelemetry


def _scripted_handler(responses: list[dict]):
    """Returns an httpx handler that pops one response per call."""
    idx = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        r = responses[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(responses) - 1)
        return httpx.Response(200, json=r)

    return handler


def _tool_call_response(tool_name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 30},
        },
    }


def _final_response(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
        },
    }


class FullStackIntegrationTests(unittest.TestCase):
    def test_complete_run_with_all_production_components(self):
        # Two-turn agent: tool call -> final answer.
        responses = [
            _tool_call_response("read_doc", {"path": "alice@example.com"}),
            _final_response("Document read successfully for [REDACTED_EMAIL]."),
        ]

        with tempfile.TemporaryDirectory() as d:
            ckpath = Path(d) / "chk.db"
            tracepath = Path(d) / "trace.jsonl"

            model = OpenAICompatibleModel(
                api_key="test-key",
                base_url="https://mock/v1",
                model="gpt-4o-mini",
                client=httpx.AsyncClient(transport=httpx.MockTransport(_scripted_handler(responses))),
            )

            def read_doc(args, ctx):
                # Returns content that contains PII; redactor will mask it later.
                return ToolResult(
                    tool_name="read_doc",
                    ok=True,
                    output=f"Hello, contact alice@example.com",
                )

            harness = Harness(
                tools=[Tool.from_sync(
                    name="read_doc",
                    description="read a document",
                    handler=read_doc,
                    required_permissions=frozenset({"fs.read"}),
                )],
                granted_permissions=frozenset({"fs.read"}),
                validator=lambda s: ValidationResult(),
            )

            ck = SqliteCheckpointer(ckpath)
            telemetry = JsonlTelemetry(tracepath, redactor=RegexRedactor())
            runtime = Runtime.production(
                model=model,
                checkpointer=ck,
                telemetry=telemetry,
                compactor=TokenAwareCompactor(max_tokens=100_000),
                max_iterations=5,
                max_cost_usd=1.0,  # generous budget
                cost_estimator=lambda u: estimate_cost(
                    u.model_id, u.prompt_tokens, u.completion_tokens, u.cached_tokens
                ),
            )

            result = asyncio.run(runtime.arun(harness, "Read the document"))

            # 1. Run completed
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertIn("Document", result.summary or "")

            # 2. Cost was tracked
            self.assertGreater(result.state.metadata.get("cost_usd", 0), 0.0)
            tokens = result.state.metadata.get("tokens", {})
            self.assertGreater(tokens.get("prompt", 0), 0)
            self.assertGreater(tokens.get("completion", 0), 0)
            self.assertGreater(tokens.get("cached", 0), 0)

            # 3. Checkpoint can persist and reload
            asyncio.run(ck.asave(result.state.run_id, result.state))
            loaded = asyncio.run(ck.aload(result.state.run_id))
            self.assertEqual(loaded.run_id, result.state.run_id)
            self.assertEqual(
                loaded.metadata["cost_usd"],
                result.state.metadata["cost_usd"],
            )
            ck.close()

            # 4. Telemetry was written; tool_args / outputs are not logged by
            # default (design decision: avoid leaking PII through telemetry
            # without explicit opt-in). Redaction itself is exercised in the
            # security suite.
            telemetry.close()
            lines = tracepath.read_text().strip().splitlines()
            self.assertGreater(len(lines), 0)
            content = "\n".join(lines)
            # Sensitive content from tool output must not be in telemetry
            self.assertNotIn("alice@example.com", content)
            # We expect at least one decision + one tool lifecycle event
            self.assertIn('"event": "decision"', content)
            self.assertIn('"event": "tool_completed"', content)

    def test_approval_pause_then_resume_in_new_runtime(self):
        """User-facing scenario: long-running run pauses on approval, server
        restarts, resumed cleanly with the same checkpoint."""

        responses_phase1 = [_tool_call_response("send_email", {"to": "ceo@acme.com"})]
        responses_phase2 = [_final_response("Email sent to CEO.")]

        def make_runtime(responses, ckpath, tracepath):
            model = OpenAICompatibleModel(
                api_key="test",
                base_url="https://mock/v1",
                model="gpt-4o-mini",
                client=httpx.AsyncClient(transport=httpx.MockTransport(_scripted_handler(responses))),
            )
            ck = SqliteCheckpointer(ckpath)
            telemetry = JsonlTelemetry(tracepath)
            rt = Runtime.production(
                model=model,
                checkpointer=ck,
                telemetry=telemetry,
            )
            return rt, ck, telemetry

        def email_tool(args, ctx):
            return ToolResult(tool_name="send_email", ok=True, output="sent")

        def _email_was_sent(state) -> bool:
            for msg in state.messages:
                for block in msg.content:
                    if isinstance(block, ToolUseBlock) and block.name == "send_email":
                        return True
            return False

        harness = Harness(
            tools=[Tool.from_sync(name="send_email", description="send", handler=email_tool)],
            guard=lambda d, s: GuardResult(requires_approval=True, reason="CEO email"),
            validator=lambda s: ValidationResult(
                done=_email_was_sent(s),
                summary="email sent",
            ),
        )

        with tempfile.TemporaryDirectory() as d:
            ckpath = Path(d) / "chk.db"
            tracepath = Path(d) / "trace.jsonl"

            # Phase 1: server A runs to pause
            rt_a, ck_a, tel_a = make_runtime(responses_phase1, ckpath, tracepath)
            paused = asyncio.run(rt_a.arun(harness, "send approval email"))
            self.assertEqual(paused.status, RunStatus.APPROVAL_REQUIRED)
            asyncio.run(ck_a.asave(paused.state.run_id, paused.state))
            ck_a.close()
            tel_a.close()

            # Phase 2: server B (new process simulation) loads + resumes
            rt_b, ck_b, tel_b = make_runtime(responses_phase2, ckpath, tracepath)
            loaded = asyncio.run(ck_b.aload(paused.state.run_id))
            self.assertIsNotNone(loaded)
            resumed = asyncio.run(rt_b.aresume(harness, loaded, approved=True))
            ck_b.close()
            tel_b.close()

            self.assertEqual(resumed.status, RunStatus.COMPLETED)
            self.assertEqual(resumed.summary, "email sent")


if __name__ == "__main__":
    unittest.main()
