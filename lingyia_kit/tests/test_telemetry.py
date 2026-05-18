"""Tests for the production telemetry sinks.

Covers:
- StructuredLogTelemetry: JSON shape, redactor invocation, doesn't double-log
- JsonlTelemetry: file append, concurrent writes
- OTelTelemetrySink: span name, attributes, error status, duration mapping
  via an injected fake tracer (no real OTel dep)
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import tempfile
import threading
import unittest
from pathlib import Path

from lingyia_core import TelemetryEvent
from lingyia_kit.telemetry import (
    JsonlTelemetry,
    OTelTelemetrySink,
    StructuredLogTelemetry,
)


def _event(kind: str = "tool_completed", **payload) -> TelemetryEvent:
    return TelemetryEvent(
        kind=kind,
        iteration=payload.pop("iteration", 1),
        run_id=payload.pop("run_id", "r-test"),
        payload=payload,
    )


# StructuredLog -----------------------------------------------------------


class StructuredLogTelemetryTests(unittest.TestCase):
    def _capture_logger(self) -> tuple[logging.Logger, io.StringIO]:
        buf = io.StringIO()
        logger = logging.getLogger(f"test-{id(buf)}")
        logger.handlers.clear()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger, buf

    def test_emits_one_json_line_per_event(self):
        logger, buf = self._capture_logger()
        sink = StructuredLogTelemetry(logger=logger)
        sink.emit(_event(tool="read_file", duration_ms=10, ok=True))
        line = buf.getvalue().strip()
        rec = json.loads(line)
        self.assertEqual(rec["event"], "tool_completed")
        self.assertEqual(rec["iteration"], 1)
        self.assertEqual(rec["run_id"], "r-test")
        self.assertEqual(rec["tool"], "read_file")
        self.assertEqual(rec["duration_ms"], 10)
        self.assertTrue(rec["ok"])
        # Should not produce more than one line
        self.assertEqual(buf.getvalue().count("\n"), 1)

    def test_redactor_runs_before_emit(self):
        logger, buf = self._capture_logger()
        seen = []

        def redact(p):
            seen.append(dict(p))
            return {**p, "secret": "***"}

        sink = StructuredLogTelemetry(logger=logger, redactor=redact)
        sink.emit(_event(tool="x", secret="abc123"))
        rec = json.loads(buf.getvalue().strip())
        self.assertEqual(rec["secret"], "***")
        self.assertEqual(seen[0]["secret"], "abc123")


# JSONL -------------------------------------------------------------------


class JsonlTelemetryTests(unittest.TestCase):
    def test_appends_one_line_per_event(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.jsonl"
            sink = JsonlTelemetry(path)
            sink.emit(_event("decision", kind_v="call_tool"))
            sink.emit(_event("tool_completed", tool="x", ok=True, duration_ms=5))
            sink.close()
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            r1 = json.loads(lines[0])
            r2 = json.loads(lines[1])
            self.assertEqual(r1["event"], "decision")
            self.assertEqual(r2["payload"]["tool"], "x")
            self.assertEqual(r2["payload"]["ok"], True)

    def test_concurrent_writes_dont_interleave(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.jsonl"
            sink = JsonlTelemetry(path)

            def write_many(thread_id):
                for i in range(50):
                    sink.emit(_event(
                        kind=f"e{thread_id}",
                        iteration=i,
                        run_id=f"r{thread_id}",
                    ))

            threads = [threading.Thread(target=write_many, args=(t,)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            sink.close()

            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 200)
            # Each line must be valid JSON
            for ln in lines:
                rec = json.loads(ln)
                self.assertIn("event", rec)
                self.assertIn("run_id", rec)


# OTel --------------------------------------------------------------------


class _FakeSpan:
    def __init__(self) -> None:
        self.name = ""
        self.attributes: dict = {}
        self.start_time = None
        self.end_time = None
        self.ended = False
        self.status = None

    def set_status(self, status) -> None:
        self.status = status

    def end(self, end_time=None) -> None:
        self.end_time = end_time
        self.ended = True


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_span(self, name, attributes=None, start_time=None):
        span = _FakeSpan()
        span.name = name
        span.attributes = dict(attributes or {})
        span.start_time = start_time
        self.spans.append(span)
        return span


class _FakeStatus:
    def __init__(self, code, description=""):
        self.code = code
        self.description = description


class _FakeStatusCode:
    OK = "OK"
    ERROR = "ERROR"


class _FakeTraceModule:
    Status = _FakeStatus
    StatusCode = _FakeStatusCode


class OTelTelemetrySinkTests(unittest.TestCase):
    def _sink(self) -> tuple[OTelTelemetrySink, _FakeTracer]:
        tracer = _FakeTracer()
        sink = OTelTelemetrySink(tracer=tracer, trace_module=_FakeTraceModule())
        return sink, tracer

    def test_emit_creates_span_with_run_id_and_iteration(self):
        sink, tracer = self._sink()
        sink.emit(_event(tool="read_file", duration_ms=12.4, ok=True))
        self.assertEqual(len(tracer.spans), 1)
        s = tracer.spans[0]
        self.assertEqual(s.name, "tool_completed")
        self.assertEqual(s.attributes["agent.iteration"], 1)
        self.assertEqual(s.attributes["agent.run_id"], "r-test")
        self.assertEqual(s.attributes["agent.tool"], "read_file")
        self.assertTrue(s.attributes["agent.ok"])
        self.assertTrue(s.ended)
        # duration_ms 12.4 → end_time = start_time + 12.4ms in nanoseconds
        self.assertIsNotNone(s.end_time)
        self.assertEqual(s.end_time - s.start_time, int(12.4 * 1e6))

    def test_error_payload_sets_error_status(self):
        sink, tracer = self._sink()
        sink.emit(_event("tool_error", tool="flaky", error="boom"))
        s = tracer.spans[0]
        self.assertIsNotNone(s.status)
        self.assertEqual(s.status.code, "ERROR")
        self.assertIn("boom", s.status.description)

    def test_complex_payload_values_are_repr_truncated(self):
        sink, tracer = self._sink()
        big = {"big_list": list(range(500))}
        sink.emit(_event("custom", **big))
        s = tracer.spans[0]
        val = s.attributes["agent.big_list"]
        self.assertIsInstance(val, str)
        self.assertLessEqual(len(val), 512)

    def test_no_duration_means_immediate_end(self):
        sink, tracer = self._sink()
        sink.emit(_event("decision"))  # no duration_ms
        s = tracer.spans[0]
        self.assertTrue(s.ended)
        self.assertIsNone(s.end_time)


# Runtime integration -----------------------------------------------------


class TelemetryIntegrationTests(unittest.TestCase):
    """Verify Runtime stamps run_id into every emitted event."""

    def test_runtime_stamps_run_id_on_every_event(self):
        from lingyia_core import (
            BlockKind,
            Decision,
            Harness,
            ModelCapabilities,
            Runtime,
            Tool,
            ToolResult,
            ToolUseBlock,
            ValidationResult,
        )

        fake_caps = ModelCapabilities(
            model_id="fake",
            accepts=frozenset({
                BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT,
            }),
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
        )

        captured: list[TelemetryEvent] = []

        class CaptureSink:
            def emit(self, event):
                captured.append(event)

        class OneTurnModel:
            capabilities = fake_caps

            async def adecide(self, context, state, tools):
                return Decision.call_tools([
                    ToolUseBlock(id="call-1", name="noop", input={}),
                ])

        def noop(args, ctx):
            return ToolResult(tool_name="noop", ok=True, output="x")

        harness = Harness(
            tools=[Tool.from_sync(name="noop", description="n", handler=noop)],
            validator=lambda s: ValidationResult(done=True, summary="ok"),
        )
        rt = Runtime.dev(model=OneTurnModel())
        rt.telemetry = CaptureSink()
        result = asyncio.run(rt.arun(harness, "g"))

        self.assertTrue(captured, "no telemetry was emitted")
        run_id = result.state.run_id
        for ev in captured:
            self.assertEqual(
                ev.run_id, run_id, f"event {ev.kind} missing run_id"
            )


if __name__ == "__main__":
    unittest.main()
