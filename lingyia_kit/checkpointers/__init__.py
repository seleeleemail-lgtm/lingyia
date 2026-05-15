"""Production-grade Checkpointer implementations.

``SqliteCheckpointer`` is zero-dependency (stdlib ``sqlite3``) and persists
``RunState`` to disk. Use it for single-instance deployments or local dev/test.
Multi-instance production should use a future ``PostgresCheckpointer``.
"""
from .sqlite import SqliteCheckpointer

__all__ = ["SqliteCheckpointer"]
