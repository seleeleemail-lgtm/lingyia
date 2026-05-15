"""In-memory checkpointer for dev/test only.

Production deployments must inject a durable Checkpointer (Redis, SQL, blob store).
This implementation is intentionally lossy across process restarts.
"""
from __future__ import annotations

import copy
from typing import Optional

from ..state import RunState


class InMemoryCheckpointer:
    """Holds RunState in a process-local dict. NOT durable."""

    def __init__(self) -> None:
        self._store: dict[str, RunState] = {}

    async def asave(self, run_id: str, state: RunState) -> None:
        self._store[run_id] = copy.deepcopy(state)

    async def aload(self, run_id: str) -> Optional[RunState]:
        snap = self._store.get(run_id)
        return copy.deepcopy(snap) if snap else None
