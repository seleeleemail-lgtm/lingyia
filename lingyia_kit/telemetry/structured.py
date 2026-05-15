"""Structured (JSON) logging via stdlib ``logging``.

Emits one JSON line per event. Zero external dependencies.

Field shape:
    {"ts": <unix>, "event": "tool_completed", "iteration": 3,
     "run_id": "...", ...payload}

Each payload key is hoisted to the top level so log search backends (Loki,
Cloudwatch Insights, ELK) can index them directly.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable, Optional, TextIO

from lingyia_core.state import TelemetryEvent


PayloadRedactor = Callable[[dict[str, Any]], dict[str, Any]]


class StructuredLogTelemetry:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        *,
        level: int = logging.INFO,
        stream: Optional[TextIO] = None,
        redactor: Optional[PayloadRedactor] = None,
    ) -> None:
        if logger is None:
            logger = logging.getLogger("lingyia_kit.telemetry")
            # Only install our handler if the logger is unconfigured. This
            # respects callers who pre-wired a different handler / formatter.
            if not logger.handlers:
                handler = logging.StreamHandler(stream or sys.stderr)
                handler.setFormatter(_PassthroughFormatter())
                logger.addHandler(handler)
                logger.setLevel(level)
                # Prevent double-printing from the root logger.
                logger.propagate = False
        self._logger = logger
        self._level = level
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
            **payload,
        }
        self._logger.log(
            self._level,
            json.dumps(record, default=str, ensure_ascii=False),
        )


class _PassthroughFormatter(logging.Formatter):
    """Formatter that emits the message verbatim (we serialize JSON ourselves)."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()
