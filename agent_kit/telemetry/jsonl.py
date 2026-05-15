"""JSONL telemetry: one JSON object per line, append-only file.

Best for local debugging or durable on-disk traces consumed by analysis tools
(``jq``, DuckDB, Pandas) after the fact. Thread-safe via an internal lock.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from agent_core.state import TelemetryEvent


PayloadRedactor = Callable[[dict[str, Any]], dict[str, Any]]


class JsonlTelemetry:
    def __init__(
        self,
        path: str | Path,
        *,
        redactor: Optional[PayloadRedactor] = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._redactor = redactor

    def emit(self, event: TelemetryEvent) -> None:
        payload = dict(event.payload)
        if self._redactor:
            payload = self._redactor(payload)
        record = {
            "ts": event.timestamp,
            "event": event.kind,
            "iteration": event.iteration,
            "run_id": event.run_id,
            "payload": payload,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self._file.write(line)
            self._file.write("\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except Exception:
                pass
