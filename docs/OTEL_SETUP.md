# OpenTelemetry Setup

The `OTelTelemetrySink` in `lingyia_kit.telemetry` emits one OTel span per agent event. To actually **send those spans somewhere**, you wire up an `opentelemetry-sdk` TracerProvider with an exporter. The sink itself is exporter-agnostic.

This guide shows the three setups we test against.

---

## Prereqs

```bash
pip install "lingyia[opentelemetry]"
# Above installs: opentelemetry-api, opentelemetry-sdk
```

Then pick an exporter package below.

---

## Setup 1 — Local Jaeger / Tempo / OTel Collector (OTLP gRPC)

The standard production pattern: every service ships spans to a local OTel Collector daemon, which fans out to Tempo / Jaeger / Datadog / etc. Lingyia doesn't care which backend you use.

```bash
pip install opentelemetry-exporter-otlp-proto-grpc
```

```python
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from lingyia_kit.telemetry import OTelTelemetrySink

# 1. Configure the global TracerProvider once at process start.
resource = Resource.create({
    "service.name": "my-agent-service",
    "service.version": "0.1.0",
})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(
    endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    insecure=True,
)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# 2. Hand the sink to the runtime.
runtime = Runtime.production(
    model=...,
    checkpointer=...,
    telemetry=OTelTelemetrySink(tracer_name="my-agent-service"),
)
```

Every `decision`, `tool_started`, `tool_completed`, `tool_timeout`, `interrupt`, `permission_denied`, `compacted` event becomes a span with these attributes:

- `agent.run_id`
- `agent.iteration`
- `agent.tool` (when applicable)
- `agent.cost_usd`, `agent.prompt_tokens`, `agent.completion_tokens` (on `decision`)
- `agent.duration_ms` (on `tool_completed`)
- `agent.error` (on failures, span status set to ERROR)

---

## Setup 2 — Langfuse (OTel-native since 2024)

Langfuse accepts standard OTLP. Same code as above, just point at their endpoint:

```bash
pip install opentelemetry-exporter-otlp-proto-http
```

```python
import base64, os
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

pk = os.environ["LANGFUSE_PUBLIC_KEY"]
sk = os.environ["LANGFUSE_SECRET_KEY"]
auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()

exporter = OTLPSpanExporter(
    endpoint="https://cloud.langfuse.com/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {auth}"},
)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
```

Langfuse will show each agent run as a trace, each event as a span. Tool calls and model calls are nested under the run's root span (assuming you start a parent span around each `runtime.arun(...)` call — see Setup 4).

---

## Setup 3 — Datadog APM

Datadog's OTel ingest accepts OTLP. Or use their native agent:

```bash
pip install opentelemetry-exporter-otlp-proto-grpc
```

```python
exporter = OTLPSpanExporter(
    endpoint="https://otlp.datadoghq.com",
    headers={"dd-api-key": os.environ["DATADOG_API_KEY"]},
)
provider.add_span_processor(BatchSpanProcessor(exporter))
```

Datadog will auto-correlate Lingyia spans with your other instrumented services if you propagate trace context.

---

## Setup 4 — Wrap each run in a root span (recommended)

For nicer trace trees, start a parent span around each `arun(...)`:

```python
from opentelemetry import trace

tracer = trace.get_tracer("my-agent-service")

async def run_with_trace(goal: str):
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("goal", goal[:200])
        result = await runtime.arun(harness, goal=goal)
        span.set_attribute("status", result.status.value)
        span.set_attribute("iterations", result.state.iteration)
        span.set_attribute("cost_usd", result.state.metadata.get("cost_usd", 0))
        return result
```

Now every event emitted by Lingyia during this run becomes a child span of `agent.run`.

---

## Cost / token attributes in queries

These attributes are searchable in any OTel backend:

| Attribute | Where it appears |
|---|---|
| `agent.run_id` | Every event |
| `agent.iteration` | Every event |
| `agent.cost_usd` | `decision` events (per-call cost) |
| `agent.cost_total_usd` | `decision` events (cumulative for the run) |
| `agent.prompt_tokens` | `decision` events |
| `agent.completion_tokens` | `decision` events |
| `agent.cached_tokens` | `decision` events |
| `agent.model_id` | `decision` events |
| `agent.tool` | `tool_*` events |
| `agent.duration_ms` | `tool_completed` events |

Example queries:

- **Langfuse / Tempo**: "show all runs where cost_total_usd > 1.0"
- **Datadog**: `service:my-agent-service @agent.cost_total_usd:>1.0`
- **Honeycomb**: `WHERE agent.cost_total_usd > 1.0`

---

## Troubleshooting

**Spans not appearing**

1. Check exporter endpoint is reachable: `curl -v $OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces`.
2. Confirm BatchSpanProcessor isn't dropping (set `OTEL_BSP_EXPORT_TIMEOUT_MILLIS=30000`).
3. Force a flush at shutdown: `provider.force_flush()`.

**Spans appear but lack Lingyia attributes**

The sink only adds attributes for known primitive types (str, bool, int, float). Complex payloads are stringified with `repr()` and truncated to 512 chars. If you need full-fidelity for one field, custom-extract it in your context builder.

**Auth errors hitting Langfuse**

The Basic auth header must be `base64("<public_key>:<secret_key>")`. The string before the colon is the **public** key.

---

## Future work

- A `lingyia_kit.telemetry.LangfuseSink` shortcut for the most common case.
- Optional batching / sampling controls exposed on `OTelTelemetrySink` for high-volume runs.
- PR welcome.
