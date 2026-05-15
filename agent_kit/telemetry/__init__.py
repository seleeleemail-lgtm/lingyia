"""Production telemetry sinks.

Layered by dependency footprint:

- ``StructuredLogTelemetry``: stdlib logging + JSON, zero external deps.
  Use when you only need searchable logs (Loki, CloudWatch, etc.).

- ``JsonlTelemetry``: append-only JSONL file. Best for local debugging or
  durable on-disk traces you analyze later.

- ``OTelTelemetrySink``: OpenTelemetry-backed, one span per event. Bridges
  to Langfuse, Datadog, Tempo, Jaeger, Honeycomb etc. through OTel exporters.
  Requires ``opentelemetry-api`` + ``opentelemetry-sdk`` to be installed.
"""
from .jsonl import JsonlTelemetry
from .otel import OTelTelemetrySink
from .structured import StructuredLogTelemetry

__all__ = [
    "JsonlTelemetry",
    "OTelTelemetrySink",
    "StructuredLogTelemetry",
]
