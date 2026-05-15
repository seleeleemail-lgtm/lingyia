"""Integration tests for SqliteCheckpointer.

These tests verify the contract that the Checkpointer Protocol promises:
- Round-trip preserves RunState exactly
- Cross-"process" resume works via a fresh checkpointer pointing at same file
- Concurrent saves don't corrupt state
- Unknown future schema versions are rejected loudly
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
    Observation,
    RunState,
    RunStatus,
    Runtime,
    Tool,
    ToolCall,
    ToolResult,
)
from lingyia_core.state import RUN_STATE_SCHEMA_VERSION, UnknownSchemaVersionError
from lingyia_kit.checkpointers import SqliteCheckpointer


class FixedDecisionModel:
    """Async model double that yields a scripted sequence of decisions."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)

    async def adecide(self, context, state, tools):
        if not self.decisions:
            raise AssertionError("scripted model exhausted")
        return self.decisions.pop(0)


def _seeded_state(run_id: str = "r1") -> RunState:
    s = RunState(goal="审一份 SaaS 合同", run_id=run_id)
    s.iteration = 3
    s.observations.append(Observation(
        iteration=1,
        kind="tool_result",
        payload={
            "tool_name": "read_contract",
            "tool_args": {"path": "/tmp/msa.docx"},
            "call_id": "call_a",
            "ok": True,
            "output": "MSA content...",
            "error": "",
            "duration_ms": 12.4,
            "attempts": 1,
        },
    ))
    s.feedback.append("client prefers shorter indemnity caps")
    s.metadata["client_id"] = "acme-corp"
    s.interrupt = Interrupt(
        reason=InterruptReason.APPROVAL,
        message="cap exceeds threshold",
        pending_decision=Decision.call_tool("apply_redline", {"clause": "8.2"}),
        iteration=3,
    )
    return s


class RoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_preserves_state(self):
        original = _seeded_state()
        snapshot = original.to_dict()
        json_str = json.dumps(snapshot, default=str)
        rehydrated = RunState.from_dict(json.loads(json_str))

        self.assertEqual(rehydrated.goal, original.goal)
        self.assertEqual(rehydrated.run_id, original.run_id)
        self.assertEqual(rehydrated.iteration, original.iteration)
        self.assertEqual(len(rehydrated.observations), len(original.observations))
        self.assertEqual(
            rehydrated.observations[0].payload["tool_args"],
            {"path": "/tmp/msa.docx"},
        )
        self.assertEqual(rehydrated.feedback, original.feedback)
        self.assertEqual(rehydrated.metadata["client_id"], "acme-corp")
        self.assertIsNotNone(rehydrated.interrupt)
        self.assertEqual(rehydrated.interrupt.reason, InterruptReason.APPROVAL)
        self.assertIsNotNone(rehydrated.interrupt.pending_decision)
        self.assertEqual(
            rehydrated.interrupt.pending_decision.kind,
            DecisionKind.CALL_TOOL,
        )
        self.assertEqual(
            rehydrated.interrupt.pending_decision.tool_calls[0].name,
            "apply_redline",
        )


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
        self.assertEqual(loaded.goal, "审一份 SaaS 合同")
        self.assertEqual(loaded.iteration, 3)
        self.assertIsNotNone(loaded.interrupt)
        # The pending decision survived process restart:
        self.assertEqual(
            loaded.interrupt.pending_decision.tool_calls[0].args,
            {"clause": "8.2"},
        )

    def test_concurrent_saves_dont_corrupt(self):
        async def go():
            ck = SqliteCheckpointer()
            await asyncio.gather(*[
                ck.asave(f"r{i}", RunState(goal=f"goal-{i}", run_id=f"r{i}"))
                for i in range(20)
            ])
            # also do 20 overlapping save+load
            await asyncio.gather(*[
                ck.asave(f"r{i}", RunState(goal=f"goal-{i}", run_id=f"r{i}", iteration=i))
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

    def test_future_schema_version_rejected(self):
        async def go():
            ck = SqliteCheckpointer()
            # Forge a too-new snapshot directly via the sync helper
            forged = {
                "schema_version": RUN_STATE_SCHEMA_VERSION + 99,
                "goal": "from the future",
                "run_id": "future-1",
                "iteration": 0,
                "observations": [],
                "feedback": [],
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

    def test_runtime_pause_save_resume_in_new_runtime(self):
        """End-to-end: Runtime A pauses, we checkpoint, Runtime B resumes."""
        approval_decision = Decision.call_tool("send_email", {"to": "ceo@acme.com"})
        final_decision = Decision.final_answer("approved and sent")

        async def go():
            ck = SqliteCheckpointer()

            def email_handler(args, ctx):
                return ToolResult(tool_name="send_email", ok=True, output="sent")

            def guard(decision, state):
                from lingyia_core import GuardResult
                return GuardResult(requires_approval=True, reason="emails to CEO need approval")

            from lingyia_core import Harness, ValidationResult
            harness = Harness(
                tools=[Tool.from_sync(name="send_email", description="send", handler=email_handler)],
                guard=guard,
                validator=lambda state: ValidationResult(
                    done=any(o.payload["tool_name"] == "send_email" for o in state.observations),
                    summary="email approved + sent",
                ),
            )

            # Runtime A: paused on approval
            rt_a = Runtime.dev(model=FixedDecisionModel(approval_decision))
            from lingyia_core.defaults.telemetry import NoopTelemetry
            rt_a.telemetry = NoopTelemetry()
            paused = await rt_a.arun(harness, goal="send approval email")
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
