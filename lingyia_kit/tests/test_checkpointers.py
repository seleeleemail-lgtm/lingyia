"""Integration tests for SqliteCheckpointer (v0.2 message wire shape).

These tests verify the contract that the Checkpointer Protocol promises:
- Round-trip preserves RunState exactly (messages, run_id, iteration, interrupt, metadata)
- Cross-"process" resume works via a fresh checkpointer pointing at same file
- Concurrent saves don't corrupt state
- Unknown future schema versions are rejected loudly
- v0.1 wire shape (schema_version=1) is rejected by RunState.from_dict at load time
- Runtime can pause -> save -> resume in a brand-new Runtime instance
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from lingyia_core import (
    Decision,
    DecisionKind,
    Interrupt,
    InterruptReason,
    ModelCapabilities,
    RunState,
    RunStatus,
    Runtime,
    Tool,
    ToolResult,
)
from lingyia_core.blocks import (
    BlockKind,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    TruncationBlock,
)
from lingyia_core.message import Message
from lingyia_core.state import RUN_STATE_SCHEMA_VERSION, UnknownSchemaVersionError
from lingyia_kit.checkpointers import SqliteCheckpointer


class FixedDecisionModel:
    """Async model double that yields a scripted sequence of decisions."""

    capabilities = ModelCapabilities(
        model_id="fake",
        accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
        emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
    )

    def __init__(self, *decisions):
        self.decisions = list(decisions)

    async def adecide(self, context, state, tools):
        if not self.decisions:
            raise AssertionError("scripted model exhausted")
        return self.decisions.pop(0)


def _seeded_state(run_id: str = "r1") -> RunState:
    """Build a representative v0.2 RunState with a multi-turn conversation,
    a tool_use / tool_result pair, an interrupt with a pending decision,
    and metadata. Exercises every block type that round-trips through JSON.
    """
    s = RunState(
        messages=[
            Message(role=Role.USER, content=(TextBlock(text="审一份 SaaS 合同"),)),
            Message(role=Role.ASSISTANT, content=(
                TextBlock(text="reading the contract"),
                ToolUseBlock(
                    id="call_a",
                    name="read_contract",
                    input={"path": "/tmp/msa.docx"},
                ),
            )),
            Message(role=Role.USER, content=(
                ToolResultBlock(tool_use_id="call_a", content="MSA content..."),
            )),
            # v0.1 used to store "feedback" separately — in v0.2 it's a user message.
            Message(role=Role.USER, content=(
                TextBlock(text="client prefers shorter indemnity caps"),
            )),
        ],
        run_id=run_id,
    )
    s.iteration = 3
    s.metadata["client_id"] = "acme-corp"
    s.interrupt = Interrupt(
        reason=InterruptReason.APPROVAL,
        message="cap exceeds threshold",
        pending_decision=Decision.call_tools([
            ToolUseBlock(id="call_b", name="apply_redline", input={"clause": "8.2"}),
        ]),
        iteration=3,
    )
    return s


class RoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_preserves_state(self):
        original = _seeded_state()
        snapshot = original.to_dict()
        # v0.2 wire shape assertions on the serialized dict itself.
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertIn("messages", snapshot)
        self.assertNotIn("goal", snapshot)
        self.assertNotIn("observations", snapshot)
        self.assertNotIn("feedback", snapshot)
        self.assertEqual(len(snapshot["messages"]), 4)
        # Spot-check block-level structure on the wire.
        self.assertEqual(snapshot["messages"][0]["role"], "user")
        self.assertEqual(snapshot["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(snapshot["messages"][1]["content"][1]["type"], "tool_use")
        self.assertEqual(snapshot["messages"][2]["content"][0]["type"], "tool_result")

        json_str = json.dumps(snapshot, default=str)
        rehydrated = RunState.from_dict(json.loads(json_str))

        self.assertEqual(rehydrated.run_id, original.run_id)
        self.assertEqual(rehydrated.iteration, original.iteration)
        self.assertEqual(len(rehydrated.messages), len(original.messages))
        self.assertEqual(rehydrated.messages[0].role, Role.USER)
        self.assertEqual(rehydrated.messages[0].content[0].text, "审一份 SaaS 合同")
        # Tool use block survived.
        tool_use = rehydrated.messages[1].content[1]
        self.assertIsInstance(tool_use, ToolUseBlock)
        self.assertEqual(tool_use.name, "read_contract")
        self.assertEqual(dict(tool_use.input), {"path": "/tmp/msa.docx"})
        # Tool result block survived with matching tool_use_id.
        tool_result = rehydrated.messages[2].content[0]
        self.assertIsInstance(tool_result, ToolResultBlock)
        self.assertEqual(tool_result.tool_use_id, "call_a")
        self.assertEqual(tool_result.content, "MSA content...")
        # Feedback-as-user-message survived.
        self.assertEqual(rehydrated.messages[3].role, Role.USER)
        self.assertEqual(
            rehydrated.messages[3].content[0].text,
            "client prefers shorter indemnity caps",
        )
        self.assertEqual(rehydrated.metadata["client_id"], "acme-corp")
        self.assertIsNotNone(rehydrated.interrupt)
        self.assertEqual(rehydrated.interrupt.reason, InterruptReason.APPROVAL)
        self.assertIsNotNone(rehydrated.interrupt.pending_decision)
        self.assertEqual(
            rehydrated.interrupt.pending_decision.kind,
            DecisionKind.CALL_TOOL,
        )
        pending_tool = rehydrated.interrupt.pending_decision.tool_calls[0]
        self.assertEqual(pending_tool.name, "apply_redline")
        self.assertEqual(dict(pending_tool.input), {"clause": "8.2"})


class SqliteCheckpointerTests(unittest.TestCase):
    def test_save_then_load_in_memory(self):
        async def go():
            ck = SqliteCheckpointer()
            await ck.asave("r1", _seeded_state("r1"))
            loaded = await ck.aload("r1")
            ck.close()
            return loaded

        loaded = asyncio.run(go())
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "r1")
        self.assertEqual(loaded.iteration, 3)
        self.assertEqual(loaded.metadata["client_id"], "acme-corp")
        self.assertEqual(len(loaded.messages), 4)
        self.assertEqual(loaded.messages[0].content[0].text, "审一份 SaaS 合同")

    def test_cross_process_resume_via_separate_checkpointer_instances(self):
        """The killer test: instance A saves to file, instance B opens same
        file and reads the state back. This simulates a process restart."""
        async def go(db_path):
            ck_a = SqliteCheckpointer(db_path)
            await ck_a.asave("run-xyz", _seeded_state("run-xyz"))
            ck_a.close()  # simulate process A exit

            ck_b = SqliteCheckpointer(db_path)
            loaded = await ck_b.aload("run-xyz")
            ck_b.close()
            return loaded

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "checkpoints.db"
            loaded = asyncio.run(go(db_path))
        self.assertIsNotNone(loaded)
        # The first user message (analog of v0.1 "goal") survived process restart.
        self.assertEqual(loaded.messages[0].role, Role.USER)
        self.assertEqual(loaded.messages[0].content[0].text, "审一份 SaaS 合同")
        self.assertEqual(loaded.iteration, 3)
        self.assertIsNotNone(loaded.interrupt)
        # The pending decision (now a ToolUseBlock) survived process restart:
        pending_tool = loaded.interrupt.pending_decision.tool_calls[0]
        self.assertEqual(pending_tool.name, "apply_redline")
        self.assertEqual(dict(pending_tool.input), {"clause": "8.2"})

    def test_concurrent_saves_dont_corrupt(self):
        def _state(i: int) -> RunState:
            return RunState(
                messages=[Message(
                    role=Role.USER,
                    content=(TextBlock(text=f"goal-{i}"),),
                )],
                run_id=f"r{i}",
                iteration=i,
            )

        async def go():
            ck = SqliteCheckpointer()
            await asyncio.gather(*[
                ck.asave(f"r{i}", RunState(
                    messages=[Message(
                        role=Role.USER,
                        content=(TextBlock(text=f"goal-{i}"),),
                    )],
                    run_id=f"r{i}",
                ))
                for i in range(20)
            ])
            # also do 20 overlapping save+load, this time with iteration=i set.
            await asyncio.gather(*[
                ck.asave(f"r{i}", _state(i))
                for i in range(20)
            ])
            results = await asyncio.gather(*[ck.aload(f"r{i}") for i in range(20)])
            ck.close()
            return results

        results = asyncio.run(go())
        self.assertEqual(len(results), 20)
        for i, r in enumerate(results):
            self.assertIsNotNone(r, f"r{i} missing")
            self.assertEqual(r.iteration, i)
            self.assertEqual(r.messages[0].content[0].text, f"goal-{i}")

    def test_future_schema_version_rejected(self):
        async def go():
            ck = SqliteCheckpointer()
            # Forge a too-new snapshot directly via the sync helper.
            # SqliteCheckpointer.aload short-circuits with UnknownSchemaVersionError
            # when stored version > runtime max, before reaching RunState.from_dict.
            forged = {
                "schema_version": RUN_STATE_SCHEMA_VERSION + 99,
                "messages": [],
                "run_id": "future-1",
                "iteration": 0,
                "trace": [],
                "interrupt": None,
                "metadata": {},
            }
            await asyncio.to_thread(
                ck._save_sync,
                "future-1",
                json.dumps(forged),
                forged["schema_version"],
            )
            with self.assertRaises(UnknownSchemaVersionError):
                await ck.aload("future-1")
            ck.close()

        asyncio.run(go())

    def test_sqlite_rejects_v01_wire_shape(self):
        """If a v0.1 snapshot dict is stored directly (no migration), aload
        propagates UnknownSchemaVersionError from RunState.from_dict.

        SqliteCheckpointer storage is JSON-opaque — it will happily round-trip
        any payload. The rejection happens inside RunState.from_dict, which
        refuses schema_version != RUN_STATE_SCHEMA_VERSION (i.e. != 2).
        """
        async def go():
            ck = SqliteCheckpointer()
            # Storage schema is (run_id, schema_version, state_json, ...) — see
            # lingyia_kit/checkpointers/sqlite.py:_save_sync. We bypass asave so
            # we can inject a v0.1-shaped payload that asave() would never produce.
            v01_state = {
                "schema_version": 1,
                "goal": "old goal",
                "run_id": "v01",
                "iteration": 0,
                "observations": [],
                "feedback": [],
                "trace": [],
                "interrupt": None,
                "metadata": {},
            }
            await asyncio.to_thread(
                ck._save_sync,
                "v01",
                json.dumps(v01_state),
                v01_state["schema_version"],
            )
            with self.assertRaises(UnknownSchemaVersionError):
                await ck.aload("v01")
            ck.close()

        asyncio.run(go())

    def test_truncation_block_round_trip_via_sqlite_checkpointer(self):
        """Spec §12 test #18 (codex P2.12): real SqliteCheckpointer asave/aload
        preserves TruncationBlock through JSON encoding, schema storage, and
        load behavior."""
        async def go(db_path):
            ck = SqliteCheckpointer(db_path)
            original = RunState(
                messages=[
                    Message(role=Role.SYSTEM, content=(TruncationBlock(count=17),)),
                    Message(role=Role.USER, content=(TextBlock(text="continue"),)),
                ],
                run_id="t-sqlite-roundtrip",
                iteration=5,
            )
            await ck.asave(original.run_id, original)
            loaded = await ck.aload("t-sqlite-roundtrip")
            ck.close()
            return loaded

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "checkpoints.db"
            loaded = asyncio.run(go(db_path))

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "t-sqlite-roundtrip")
        self.assertEqual(loaded.iteration, 5)
        self.assertEqual(len(loaded.messages), 2)

        marker = loaded.messages[0].content[0]
        self.assertIsInstance(marker, TruncationBlock)
        self.assertEqual(marker.count, 17)

        user_text = loaded.messages[1].content[0]
        self.assertIsInstance(user_text, TextBlock)
        self.assertEqual(user_text.text, "continue")

    def test_runtime_pause_save_resume_in_new_runtime(self):
        """End-to-end: Runtime A pauses, we checkpoint, Runtime B resumes.

        Runtime-layer integration test. Validates that a paused RunState
        produced by Runtime.arun round-trips through SqliteCheckpointer and
        can be consumed by a brand-new Runtime via aresume.
        """
        approval_decision = Decision.call_tools([
            ToolUseBlock(id="call_send", name="send_email", input={"to": "ceo@acme.com"}),
        ])
        final_decision = Decision.final_answer("approved and sent")

        async def go():
            ck = SqliteCheckpointer()

            def email_handler(args, ctx):
                return ToolResult(tool_name="send_email", ok=True, output="sent")

            def guard(decision, state):
                from lingyia_core import GuardResult
                return GuardResult(requires_approval=True, reason="emails to CEO need approval")

            from lingyia_core import Harness, ValidationResult

            def _validator(state):
                # v0.2: tool execution recorded as ToolResultBlock inside messages,
                # not as Observation. Done when any ToolResultBlock for send_email exists.
                sent = False
                for msg in state.messages:
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            # Cross-ref: find the matching ToolUseBlock for tool_name.
                            for other in state.messages:
                                for ob in other.content:
                                    if (
                                        isinstance(ob, ToolUseBlock)
                                        and ob.id == block.tool_use_id
                                        and ob.name == "send_email"
                                    ):
                                        sent = True
                return ValidationResult(done=sent, summary="email approved + sent")

            harness = Harness(
                tools=[Tool.from_sync(name="send_email", description="send", handler=email_handler)],
                guard=guard,
                validator=_validator,
            )

            # Runtime A: paused on approval
            rt_a = Runtime.dev(model=FixedDecisionModel(approval_decision))
            from lingyia_core.defaults.telemetry import NoopTelemetry
            rt_a.telemetry = NoopTelemetry()
            paused = await rt_a.arun(harness, "send approval email")
            self.assertEqual(paused.status, RunStatus.APPROVAL_REQUIRED)

            # Save snapshot
            await ck.asave(paused.state.run_id, paused.state)

            # Simulate process restart by creating a fresh Checkpointer reading the same db
            # (Already validated cross-process resume above; here we use the same connection.)
            restored = await ck.aload(paused.state.run_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.interrupt.reason, InterruptReason.APPROVAL)

            # Runtime B: brand new runtime, brand new model script
            rt_b = Runtime.dev(model=FixedDecisionModel(final_decision))
            rt_b.telemetry = NoopTelemetry()
            resumed = await rt_b.aresume(harness, restored, approved=True)
            ck.close()
            return resumed

        resumed = asyncio.run(go())
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertEqual(resumed.summary, "email approved + sent")


if __name__ == "__main__":
    unittest.main()
