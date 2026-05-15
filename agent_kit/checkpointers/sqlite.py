"""SQLite-backed Checkpointer.

Zero external dependencies — uses Python's stdlib ``sqlite3``. Sync database
calls are offloaded to a thread via ``asyncio.to_thread`` so the runtime's
event loop stays unblocked.

When to use:
- Single-instance dev/staging/prod
- Local file-based durability
- Cross-process pause/resume on one machine

When NOT to use:
- Multi-instance (use a future PostgresCheckpointer instead)
- High write throughput (>1k ops/s)
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from agent_core.state import RUN_STATE_SCHEMA_VERSION, RunState, UnknownSchemaVersionError


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id          TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    state_json      TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_updated_at
    ON agent_runs(updated_at);
"""


class SqliteCheckpointer:
    """Persist RunState snapshots to a SQLite file.

    Thread-safe: writes are serialized by an internal lock so concurrent
    ``asave`` calls from the same process don't interleave.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        # check_same_thread=False because we use a single connection across
        # threads via asyncio.to_thread; access is serialized by self._lock.
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_DDL)
        self._lock = threading.Lock()

    # --- sync helpers (run inside asyncio.to_thread) -------------------

    def _save_sync(self, run_id: str, state_json: str, version: int) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_runs (run_id, schema_version, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (run_id, version, state_json, now, now),
            )

    def _load_sync(self, run_id: str) -> Optional[tuple[int, str]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT schema_version, state_json FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return int(row[0]), row[1]

    def _delete_sync(self, run_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM agent_runs WHERE run_id = ?",
                (run_id,),
            )
            return cur.rowcount > 0

    def _list_sync(self, limit: int = 100) -> list[tuple[str, int, float]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, schema_version, updated_at "
                "FROM agent_runs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], int(r[1]), float(r[2])) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- Checkpointer Protocol ----------------------------------------

    async def asave(self, run_id: str, state: RunState) -> None:
        snapshot = state.to_dict()
        version = snapshot.get("schema_version", RUN_STATE_SCHEMA_VERSION)
        payload = json.dumps(snapshot, ensure_ascii=False, default=str)
        await asyncio.to_thread(self._save_sync, run_id, payload, version)

    async def aload(self, run_id: str) -> Optional[RunState]:
        row = await asyncio.to_thread(self._load_sync, run_id)
        if row is None:
            return None
        version, payload = row
        if version > RUN_STATE_SCHEMA_VERSION:
            raise UnknownSchemaVersionError(
                f"stored schema_version={version} exceeds runtime max "
                f"{RUN_STATE_SCHEMA_VERSION}"
            )
        data = json.loads(payload)
        return RunState.from_dict(data)

    async def adelete(self, run_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, run_id)

    async def alist_recent(self, limit: int = 100) -> list[tuple[str, int, float]]:
        """Return ``[(run_id, schema_version, updated_at), ...]`` newest first."""
        return await asyncio.to_thread(self._list_sync, limit)
