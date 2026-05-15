"""Telemetry sinks for dev/test.

Production should wire an OTel/Datadog/in-house sink. Do not ship NoopTelemetry
to production unless cost+latency observability is provided elsewhere.
"""
from __future__ import annotations

import sys
from typing import Optional, TextIO

from ..state import TelemetryEvent


class NoopTelemetry:
    """Discards all events. Tests only."""

    def emit(self, event: TelemetryEvent) -> None:
        return None


class StdoutTelemetry:
    """Prints events to stdout. Suitable for dev/debug only."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self._stream = stream or sys.stdout

    def emit(self, event: TelemetryEvent) -> None:
        self._stream.write(
            f"[telemetry] kind={event.kind} iter={event.iteration} payload={dict(event.payload)}\n"
        )
        self._stream.flush()
