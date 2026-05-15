"""OpenTelemetry-backed telemetry sink.

Lazily imports the OTel SDK so this module is safe to import even when OTel
isn't installed. ``OTelTelemetrySink(...)`` raises a clear ImportError if the
caller actually tries to use it without the dependency.

Each emitted ``TelemetryEvent`` becomes a span whose name is the event kind.
The event's ``timestamp`` is used as the span start time. If the payload
carries ``duration_ms`` (tool results do), that's used to set span end time.

Configure the global OTel ``TracerProvider`` out-of-band — for example with
``opentelemetry.sdk.trace.TracerProvider`` + an OTLP exporter via the
``OTEL_EXPORTER_OTLP_ENDPOINT`` env var. This sink just emits spans; routing
to Langfuse / Datadog / Jaeger / Tempo is the OTel SDK's job.
"""
from __future__ import annotations

from typing import Any, Optional

from agent_core.state import TelemetryEvent


_PRIMITIVE_TYPES = (str, bool, int, float)
_MAX_ATTR_REPR = 512


class OTelTelemetrySink:
    """Emit each TelemetryEvent as a short-lived OTel span."""

    def __init__(
        self,
        tracer_name: str = "agent_core",
        *,
        tracer: Optional[Any] = None,
        trace_module: Optional[Any] = None,
    ) -> None:
        if tracer is None or trace_module is None:
            try:
                from opentelemetry import trace as _trace
            except ImportError as exc:
                raise ImportError(
                    "OTelTelemetrySink requires the OpenTelemetry SDK. "
                    "Install with: "
                    "`pip install opentelemetry-api opentelemetry-sdk`"
                ) from exc
            self._trace = trace_module or _trace
            self._tracer = tracer or _trace.get_tracer(tracer_name)
        else:
            # Test path: caller injects a tracer + the OTel-like ``trace``
            # module (for Status/StatusCode access). Lets us unit-test the
            # sink without OTel actually installed.
            self._trace = trace_module
            self._tracer = tracer

    def emit(self, event: TelemetryEvent) -> None:
        attributes: dict[str, Any] = {
            "agent.iteration": event.iteration,
            "agent.run_id": event.run_id,
        }
        for k, v in event.payload.items():
            if v is None:
                continue
            if isinstance(v, _PRIMITIVE_TYPES):
                attributes[f"agent.{k}"] = v
            else:
                attributes[f"agent.{k}"] = repr(v)[:_MAX_ATTR_REPR]

        duration_ms = float(event.payload.get("duration_ms", 0) or 0)
        start_ns = int(event.timestamp * 1e9)
        end_ns: Optional[int] = (
            start_ns + int(duration_ms * 1e6) if duration_ms > 0 else None
        )

        span = self._tracer.start_span(
            event.kind,
            attributes=attributes,
            start_time=start_ns,
        )
        err = event.payload.get("error")
        if err:
            try:
                Status = self._trace.Status
                StatusCode = self._trace.StatusCode
                span.set_status(Status(StatusCode.ERROR, str(err)))
            except (AttributeError, TypeError):
                # Some OTel versions / mocks expose a flatter API. Best-effort.
                pass
        span.end(end_time=end_ns)
