"""Postgres-backed Checkpointer.

Best fit for multi-instance production: any number of agent workers can save
and load run state through a single Postgres cluster with full ACID semantics
and JSONB indexing.

Requires the ``asyncpg`` package: ``pip install lingyia[postgres]``.

Tested against Postgres 14+. The schema uses ``JSONB`` for the state column
so you can run ad-hoc queries on agent metadata (cost, tokens, iteration)
directly in SQL.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from lingyia_core.state import RUN_STATE_SCHEMA_VERSION, RunState, UnknownSchemaVersionError


_SCHEMA_DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    run_id          TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    state_json      JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_{table}_updated_at
    ON {table} (updated_at DESC);
"""


class PostgresCheckpointer:
    """RunState persistence backed by Postgres + asyncpg.

    Multiple workers can share one Checkpointer instance (it holds a connection
    pool). The pool is created lazily on the first operation so construction
    is cheap and doesn't require a running event loop.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "lingyia_runs",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if not dsn:
            raise ValueError("dsn is required")
        # Basic SQL-injection defense for table name. Only allow alphanumerics + underscore.
        if not table.replace("_", "").isalnum():
            raise ValueError(f"invalid table name: {table!r}")
        self._dsn = dsn
        self._table = table
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: Optional[Any] = None
        # asyncpg is imported lazily on first use. This keeps construction
        # cheap and makes the class safe to instantiate in tests even when
        # asyncpg isn't installed (tests inject ``self._asyncpg`` directly).
        self._asyncpg: Optional[Any] = None

    def _load_asyncpg(self) -> Any:
        if self._asyncpg is not None:
            return self._asyncpg
        try:
            import asyncpg  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PostgresCheckpointer requires asyncpg. "
                "Install with: pip install 'lingyia[postgres]'"
            ) from exc
        self._asyncpg = asyncpg
        return asyncpg

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            asyncpg = self._load_asyncpg()
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
            async with self._pool.acquire() as conn:
                await conn.execute(_SCHEMA_DDL_TEMPLATE.format(table=self._table))
        return self._pool

    async def asave(self, run_id: str, state: RunState) -> None:
        snapshot = state.to_dict()
        version = snapshot.get("schema_version", RUN_STATE_SCHEMA_VERSION)
        payload = json.dumps(snapshot, ensure_ascii=False, default=str)
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table}
                    (run_id, schema_version, state_json, updated_at)
                VALUES ($1, $2, $3::jsonb, NOW())
                ON CONFLICT (run_id) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    state_json     = EXCLUDED.state_json,
                    updated_at     = NOW()
                """,
                run_id,
                version,
                payload,
            )

    async def aload(self, run_id: str) -> Optional[RunState]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT schema_version, state_json FROM {self._table} WHERE run_id = $1",
                run_id,
            )
        if row is None:
            return None
        version = int(row["schema_version"])
        if version > RUN_STATE_SCHEMA_VERSION:
            raise UnknownSchemaVersionError(
                f"stored schema_version={version} exceeds runtime max "
                f"{RUN_STATE_SCHEMA_VERSION}"
            )
        # asyncpg may return JSONB as a Python object already, or as a string
        # depending on codec configuration.
        raw = row["state_json"]
        if isinstance(raw, (dict, list)):
            data = raw
        else:
            data = json.loads(raw)
        return RunState.from_dict(data)

    async def adelete(self, run_id: str) -> bool:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table} WHERE run_id = $1", run_id
            )
        # asyncpg's execute returns "DELETE <n>"
        return result.startswith("DELETE ") and result != "DELETE 0"

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
