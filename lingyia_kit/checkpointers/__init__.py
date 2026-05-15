"""Production-grade Checkpointer implementations.

Three batteries:

- ``SqliteCheckpointer``: zero-dep, single-instance, on-disk durability.
  Best for dev / staging / single-server prod.
- ``PostgresCheckpointer``: multi-instance, queryable JSONB, full ACID.
  Best for multi-worker production. Requires ``lingyia[postgres]``.
- ``RedisCheckpointer``: fast, optional TTL auto-expiry. Best when you need
  ephemeral pause/resume across workers. Requires ``lingyia[redis]``.
"""
from .sqlite import SqliteCheckpointer

__all__ = ["SqliteCheckpointer"]

# Optional checkpointers — only export when their dep is installed.
try:
    from .postgres import PostgresCheckpointer  # noqa: F401
    __all__.append("PostgresCheckpointer")
except ImportError:
    pass

try:
    from .redis import RedisCheckpointer  # noqa: F401
    __all__.append("RedisCheckpointer")
except ImportError:
    pass
