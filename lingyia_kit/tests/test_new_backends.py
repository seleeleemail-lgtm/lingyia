"""Tests for the new v0.1.0 production backends:
PostgresCheckpointer, RedisCheckpointer, sub_agent_tool.

These tests use mock backends (no real Postgres or Redis required). Real
integration testing happens in CI or by running against actual services.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

from lingyia_core import (
    BlockKind,
    Decision,
    Harness,
    Message,
    ModelCapabilities,
    Role,
    RunState,
    RunStatus,
    Runtime,
    TextBlock,
    Tool,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ValidationResult,
)
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_core.state import RUN_STATE_SCHEMA_VERSION


_FAKE_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
    emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
)


# Helpers ----------------------------------------------------------------


def _seeded_state(run_id: str = "r1") -> RunState:
    state = RunState(
        messages=[Message(role=Role.USER, content=(TextBlock(text="审一份合同"),))],
        run_id=run_id,
        iteration=2,
    )
    state.metadata["client_id"] = "acme"
    state.metadata["cost_usd"] = 0.05
    return state


# Postgres ---------------------------------------------------------------


class _MockPostgresConn:
    """Minimal asyncpg-shaped connection mock."""

    def __init__(self, store: dict):
        self._store = store

    async def execute(self, query: str, *args: Any) -> str:
        if query.lstrip().upper().startswith("CREATE TABLE"):
            return "CREATE TABLE"
        if "INSERT INTO" in query and "ON CONFLICT" in query:
            run_id, version, payload = args
            self._store[run_id] = {"v": int(version), "s": payload}
            return "INSERT 0 1"
        if query.lstrip().upper().startswith("DELETE"):
            (run_id,) = args
            deleted = self._store.pop(run_id, None)
            return "DELETE 1" if deleted else "DELETE 0"
        return ""

    async def fetchrow(self, query: str, *args: Any) -> Optional[dict]:
        if "SELECT schema_version" in query:
            (run_id,) = args
            row = self._store.get(run_id)
            if row is None:
                return None
            return {"schema_version": row["v"], "state_json": row["s"]}
        return None


class _MockPostgresPoolCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _MockPostgresPool:
    def __init__(self):
        self._store: dict = {}
        self._conn = _MockPostgresConn(self._store)

    def acquire(self):
        return _MockPostgresPoolCM(self._conn)

    async def close(self):
        return None


class _MockAsyncpgModule:
    def __init__(self):
        self.pool = _MockPostgresPool()
        self.created_with: Optional[dict] = None

    async def create_pool(self, dsn: str, **kwargs: Any) -> _MockPostgresPool:
        self.created_with = {"dsn": dsn, **kwargs}
        return self.pool


class PostgresCheckpointerTests(unittest.TestCase):
    def _make_checkpointer(self):
        from lingyia_kit.checkpointers.postgres import PostgresCheckpointer
        ck = PostgresCheckpointer(dsn="postgres://test/db")
        mock = _MockAsyncpgModule()
        ck._asyncpg = mock
        return ck, mock

    def test_save_then_load_round_trip(self):
        ck, _mock = self._make_checkpointer()

        async def go():
            state = _seeded_state("r-pg-1")
            await ck.asave(state.run_id, state)
            loaded = await ck.aload(state.run_id)
            await ck.aclose()
            return loaded

        loaded = asyncio.run(go())
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "r-pg-1")
        self.assertEqual(loaded.metadata["client_id"], "acme")
        self.assertAlmostEqual(loaded.metadata["cost_usd"], 0.05)
        self.assertEqual(loaded.iteration, 2)

    def test_aload_missing_returns_none(self):
        ck, _mock = self._make_checkpointer()
        loaded = asyncio.run(ck.aload("does-not-exist"))
        self.assertIsNone(loaded)

    def test_invalid_table_name_rejected(self):
        from lingyia_kit.checkpointers.postgres import PostgresCheckpointer
        with self.assertRaises(ValueError):
            PostgresCheckpointer(dsn="postgres://x", table="users; DROP TABLE x")

    def test_pool_created_with_dsn_and_size(self):
        from lingyia_kit.checkpointers.postgres import PostgresCheckpointer
        ck = PostgresCheckpointer(
            dsn="postgres://test/db",
            min_pool_size=2,
            max_pool_size=20,
        )
        mock = _MockAsyncpgModule()
        ck._asyncpg = mock

        async def go():
            await ck.asave("r1", _seeded_state("r1"))

        asyncio.run(go())
        self.assertEqual(mock.created_with["dsn"], "postgres://test/db")
        self.assertEqual(mock.created_with["min_size"], 2)
        self.assertEqual(mock.created_with["max_size"], 20)


# Redis ------------------------------------------------------------------


class _MockRedisClient:
    """Minimal redis-py asyncio interface mock backed by a dict."""

    def __init__(self):
        self.store: dict[str, dict] = {}
        self.expires: dict[str, int] = {}
        self.closed = False

    async def hset(self, key: str, mapping: dict) -> int:
        self.store.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def hgetall(self, key: str) -> dict:
        return dict(self.store.get(key, {}))

    async def expire(self, key: str, ttl: int) -> bool:
        self.expires[key] = ttl
        return True

    async def delete(self, key: str) -> int:
        removed = self.store.pop(key, None)
        self.expires.pop(key, None)
        return 1 if removed else 0

    async def aclose(self):
        self.closed = True


class RedisCheckpointerTests(unittest.TestCase):
    def _make_checkpointer(self, ttl: Optional[int] = None):
        from lingyia_kit.checkpointers.redis import RedisCheckpointer
        client = _MockRedisClient()
        ck = RedisCheckpointer(client=client, ttl_seconds=ttl)
        return ck, client

    def test_save_then_load(self):
        ck, client = self._make_checkpointer()

        async def go():
            await ck.asave("r-redis-1", _seeded_state("r-redis-1"))
            return await ck.aload("r-redis-1")

        loaded = asyncio.run(go())
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "r-redis-1")
        self.assertAlmostEqual(loaded.metadata["cost_usd"], 0.05)
        # Key was stored
        self.assertIn("lingyia:run:r-redis-1", client.store)

    def test_ttl_applied_when_configured(self):
        ck, client = self._make_checkpointer(ttl=3600)

        async def go():
            await ck.asave("r-ttl-1", _seeded_state("r-ttl-1"))

        asyncio.run(go())
        self.assertEqual(client.expires.get("lingyia:run:r-ttl-1"), 3600)

    def test_no_ttl_when_not_configured(self):
        ck, client = self._make_checkpointer(ttl=None)

        async def go():
            await ck.asave("r-no-ttl", _seeded_state("r-no-ttl"))

        asyncio.run(go())
        self.assertEqual(client.expires, {})

    def test_aload_missing_returns_none(self):
        ck, _ = self._make_checkpointer()
        loaded = asyncio.run(ck.aload("never-saved"))
        self.assertIsNone(loaded)

    def test_delete_returns_true_on_hit_false_on_miss(self):
        ck, _ = self._make_checkpointer()

        async def go():
            await ck.asave("r-del", _seeded_state("r-del"))
            hit = await ck.adelete("r-del")
            miss = await ck.adelete("r-del-missing")
            return hit, miss

        hit, miss = asyncio.run(go())
        self.assertTrue(hit)
        self.assertFalse(miss)

    def test_invalid_prefix_rejected(self):
        from lingyia_kit.checkpointers.redis import RedisCheckpointer
        client = _MockRedisClient()
        with self.assertRaises(ValueError):
            RedisCheckpointer(client=client, key_prefix="")
        with self.assertRaises(ValueError):
            RedisCheckpointer(client=client, ttl_seconds=0)


# Sub-agent tool ---------------------------------------------------------


class SubAgentToolTests(unittest.TestCase):
    def _build_sub_agent(self):
        """Return a (runtime, harness) that always emits one tool call then
        a final answer summarizing it."""

        class _TwoTurnModel:
            capabilities = _FAKE_CAPS

            def __init__(self):
                self.n = 0

            async def adecide(self, ctx, state, tools):
                self.n += 1
                if self.n == 1:
                    return Decision.call_tools([
                        ToolUseBlock(id="inner-1", name="inner", input={"x": "y"}),
                    ])
                return Decision.final_answer("sub-agent finished")

        def inner_handler(args, ctx):
            return ToolResult(tool_name="inner", ok=True, output="ran")

        # v0.2-α (codex verify P2): with no validator the default identity
        # is preserved by the runtime so the FINAL_ANSWER path trusts the
        # model. An explicit `lambda s: ValidationResult(done=False)`
        # would now deliberately reject and loop — exactly the case codex
        # flagged — so we leave the harness validator unset here.
        harness = Harness(
            tools=[Tool.from_sync(name="inner", description="inner tool", handler=inner_handler)],
        )
        rt = Runtime.dev(model=_TwoTurnModel(), max_iterations=4)
        rt.telemetry = NoopTelemetry()
        return rt, harness

    def test_sub_agent_completes_and_returns_summary(self):
        from lingyia_kit.tools import sub_agent_tool

        rt, harness = self._build_sub_agent()
        tool = sub_agent_tool(
            name="legal_review",
            description="Run the legal review sub-agent.",
            runtime=rt,
            harness=harness,
        )
        from lingyia_core import ToolContext
        ctx = ToolContext(run_id="parent", iteration=0, messages=(), metadata={})

        result = asyncio.run(tool.handler({"goal": "review this clause"}, ctx))
        self.assertTrue(result.ok)
        self.assertEqual(result.output["status"], "completed")
        self.assertEqual(result.output["summary"], "sub-agent finished")
        self.assertGreaterEqual(result.output["iterations"], 1)
        self.assertIn("run_id", result.output)

    def test_missing_goal_arg_fails_gracefully(self):
        from lingyia_kit.tools import sub_agent_tool

        rt, harness = self._build_sub_agent()
        tool = sub_agent_tool(
            name="legal_review",
            description="...",
            runtime=rt,
            harness=harness,
        )
        from lingyia_core import ToolContext
        ctx = ToolContext(run_id="parent", iteration=0, messages=(), metadata={})

        result = asyncio.run(tool.handler({}, ctx))
        self.assertFalse(result.ok)
        self.assertIn("missing or empty", result.error)

    def test_sub_agent_used_as_tool_in_parent_runtime(self):
        """Real composition test: parent agent calls sub-agent tool, completes."""
        from lingyia_kit.tools import sub_agent_tool

        sub_rt, sub_harness = self._build_sub_agent()
        sub_tool = sub_agent_tool(
            name="research",
            description="Run a research sub-agent.",
            runtime=sub_rt,
            harness=sub_harness,
        )

        class _ParentModel:
            capabilities = _FAKE_CAPS

            def __init__(self):
                self.n = 0

            async def adecide(self, ctx, state, tools):
                self.n += 1
                if self.n == 1:
                    return Decision.call_tools([
                        ToolUseBlock(
                            id="research-1",
                            name="research",
                            input={"goal": "find X"},
                        ),
                    ])
                return Decision.final_answer("done with research")

        parent_harness = Harness(
            tools=[sub_tool],
            # v0.2-α (codex verify P2): no explicit validator → default
            # (trust the model's FINAL_ANSWER, loop after tool calls).
        )
        parent_rt = Runtime.dev(model=_ParentModel(), max_iterations=4)
        parent_rt.telemetry = NoopTelemetry()

        result = asyncio.run(parent_rt.arun(parent_harness, "delegate research"))
        self.assertEqual(result.status, RunStatus.COMPLETED)
        # Parent should have one ToolResultBlock from the sub-agent.
        tool_results = [
            b for m in result.state.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        self.assertEqual(len(tool_results), 1)
        self.assertFalse(tool_results[0].is_error)
        # Sub-agent's run_id is surfaced in the ToolResultBlock payload.
        # The runtime serializes the dict via str(), so check the substring.
        self.assertIn("run_id", tool_results[0].content)


if __name__ == "__main__":
    unittest.main()
