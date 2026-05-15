"""Redis-backed Checkpointer.

Best fit for: speed, ephemeral runs, cross-instance pause/resume where you
don't need long-term retention. State is stored as JSON strings keyed by run id
with optional TTL.

Requires the ``redis`` package: ``pip install lingyia[redis]``.

Trade-offs vs Postgres:
- Redis: faster, simpler, no SQL queryability, optional TTL for auto-cleanup.
- Postgres: queryable JSONB, full ACID, retention via SQL, larger ops surface.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from lingyia_core.state import RUN_STATE_SCHEMA_VERSION, RunState, UnknownSchemaVersionError


_VERSION_FIELD = "v"
_PAYLOAD_FIELD = "s"


class RedisCheckpointer:
    """RunState persistence backed by Redis hashes.

    Each run lives at ``{prefix}:{run_id}`` as a Hash with two fields: schema
    version and the JSON-encoded state. Optional TTL auto-expires old runs.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "lingyia:run",
        ttl_seconds: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        if not key_prefix:
            raise ValueError("key_prefix must be a non-empty string")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0 when set")
        self._url = url
        self._prefix = key_prefix
        self._ttl = ttl_seconds
        if client is not None:
            # Test path: caller injects a redis-compatible async client.
            self._client = client
            return
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "RedisCheckpointer requires redis. "
                "Install with: pip install 'lingyia[redis]'"
            ) from exc
        self._client = aioredis.from_url(url, decode_responses=True)

    def _key(self, run_id: str) -> str:
        return f"{self._prefix}:{run_id}"

    async def asave(self, run_id: str, state: RunState) -> None:
        snapshot = state.to_dict()
        version = snapshot.get("schema_version", RUN_STATE_SCHEMA_VERSION)
        payload = json.dumps(snapshot, ensure_ascii=False, default=str)
        key = self._key(run_id)
        await self._client.hset(
            key,
            mapping={
                _VERSION_FIELD: str(version),
                _PAYLOAD_FIELD: payload,
            },
        )
        if self._ttl is not None:
            await self._client.expire(key, self._ttl)

    async def aload(self, run_id: str) -> Optional[RunState]:
        key = self._key(run_id)
        data = await self._client.hgetall(key)
        if not data:
            return None
        version = int(data.get(_VERSION_FIELD, RUN_STATE_SCHEMA_VERSION))
        if version > RUN_STATE_SCHEMA_VERSION:
            raise UnknownSchemaVersionError(
                f"stored schema_version={version} exceeds runtime max "
                f"{RUN_STATE_SCHEMA_VERSION}"
            )
        payload = data.get(_PAYLOAD_FIELD)
        if not payload:
            return None
        return RunState.from_dict(json.loads(payload))

    async def adelete(self, run_id: str) -> bool:
        n = await self._client.delete(self._key(run_id))
        return bool(n)

    async def aclose(self) -> None:
        # Some redis clients expose aclose, others close.
        for method in ("aclose", "close"):
            fn = getattr(self._client, method, None)
            if fn is not None:
                result = fn()
                if hasattr(result, "__await__"):
                    await result
                return
