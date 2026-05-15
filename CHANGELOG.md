# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Roadmap
- Inspect AI bridge runner (200+ external benchmarks)
- Reflection / replan agent pattern
- Streaming token deltas (currently event-level only)
- First-party `LangfuseSink` shortcut
- Example projects in `examples/` for legal / code / support domains

## [0.1.0] — 2026-05-15

Initial public release.

### Added (consolidated v0.1.0 scope)

#### New in the final v0.1.0 cut (post-rename additions)
- `lingyia_kit/checkpointers/postgres.py`: `PostgresCheckpointer` for
  multi-instance production. JSONB storage, lazy connection pool,
  table-name validation against SQL injection. Requires
  `pip install 'lingyia[postgres]'`.
- `lingyia_kit/checkpointers/redis.py`: `RedisCheckpointer` for fast
  ephemeral pause/resume. Hash-based storage, optional TTL auto-expiry.
  Requires `pip install 'lingyia[redis]'`.
- `lingyia_kit/tools/sub_agent.py`: `sub_agent_tool` factory wrapping a
  `Runtime + Harness` as a `Tool`. The canonical agent-as-tool pattern.
  Surfaces sub-agent cost, tokens, status, and run_id on the parent's
  observation so cross-agent observability stays intact.
- `docs/OTEL_SETUP.md`: real OpenTelemetry exporter wiring guide
  covering OTLP gRPC / HTTP, Langfuse, Datadog, with example queries.
- `README.zh.md`: full Chinese-language README.
- `pyproject.toml`: new `postgres` and `redis` extras; the `all` extra
  now installs every optional dependency.

### Added

#### lingyia_core (L1 runtime)
- `Runtime` orchestrator with `dev()` and `production()` constructors.
  Production mode fails closed if checkpointer or telemetry are missing.
- `Harness` for domain configuration: tools, validator, guard, context builder.
  Granted-permission gating for tools.
- `Decision` with four kinds (call_tool, final_answer, ask_human, abort) and
  parallel `tool_calls`. Enum-backed with str fallback for legacy callers.
- `RunState` fully JSON-serializable via `to_dict()` / `from_dict()` with
  schema versioning and `UnknownSchemaVersionError` for forward incompat.
- `Tool` with input schema, timeout, retry policy, idempotency, side-effect
  classification, and required-permissions set. `from_sync` / `from_async`
  factories so sync tools are first-class.
- `ToolExecutor` enforces per-tool timeout + retry with `RetryPolicy`. Global
  `max_concurrency` cap via semaphore. Sync tools run via `to_thread`;
  `tool_timeout` events flag "may still be running" cases.
- Five Protocols: `Model`, `Tool` (schema), `Checkpointer`, `Compactor`,
  `TelemetrySink`.
- `Runtime.astream(harness, goal)` yields telemetry events live and a final
  RunResult.
- Per-run cost tracking: `ModelUsage` on `Decision`, accumulated to
  `state.metadata["cost_usd"]`. `max_cost_usd` aborts on overspend.
  `cost_estimator` hook for adapters that only return token counts.
- All telemetry events carry `run_id` for downstream correlation.
- `asyncio.CancelledError` propagates instead of being absorbed into
  failed-result.

#### lingyia_kit (L2 production batteries)

**Adapters** (`lingyia_kit.adapters`)
- `OpenAICompatibleModel` base (pure httpx, no SDK).
- Subclasses for `OpenAIModel`, `SiliconFlowModel`, `MiniMaxModel`.
- `AnthropicModel` using the official anthropic SDK; tool_use blocks shape.
- Reconstructs multi-turn assistant tool_calls history from observations.

**Checkpointers** (`lingyia_kit.checkpointers`)
- `SqliteCheckpointer`: stdlib sqlite3 + asyncio.to_thread. WAL journaling,
  thread-safe writes, cross-process resume tested.

**Telemetry** (`lingyia_kit.telemetry`)
- `StructuredLogTelemetry`: stdlib logging + JSON lines.
- `JsonlTelemetry`: append-only file, thread-safe.
- `OTelTelemetrySink`: lazy-imports opentelemetry; one span per event.

**Compactors** (`lingyia_kit.compactors`)
- `TokenAwareCompactor` with `char_div4_estimator` (default) and
  `tiktoken_estimator(model)` lazy-import.

**Resilience** (`lingyia_kit.resilience`)
- `CircuitBreaker` (CLOSED / OPEN / HALF_OPEN).
- `TokenBucketRateLimiter` async.
- `ProtectedModel` wraps any Model with circuit + rate limit.

**Security** (`lingyia_kit.secrets`, `lingyia_kit.redaction`)
- `EnvSecretBackend`, `DotenvSecretBackend`, `InMemorySecretBackend`,
  `ChainSecretBackend`.
- `RegexRedactor` for email, phone (CN/US), credit card, CN ID card,
  sk-* API keys, plus sensitive-key overrides.

**Tools** (`lingyia_kit.tools`)
- `read_file_tool`, `write_file_tool`, `list_dir_tool` with sandbox roots.
- `fetch_url_tool` with optional host allow-list + byte cap.
- `run_shell_tool` disabled by default, opt-in allow-list gated.

**Patterns** (`lingyia_kit.patterns`)
- `react_harness` factory: default ReAct system prompt + lightweight
  context windowing.

**Eval** (`lingyia_kit.eval`)
- `bfcl_runner` for BFCL V3 single-turn function-calling accuracy.
  Supports simple / parallel / multiple splits with value-list scoring.

**Pricing** (`lingyia_kit.pricing`)
- Pricing table covering OpenAI, Anthropic, SiliconFlow, MiniMax.
- `estimate_cost()` with cached-token discount.
- `register_pricing()` for runtime overrides.

### Tests
- 92 unit + integration tests passing across Python 3.9.
- Cross-process pause/resume via two `SqliteCheckpointer` instances.
- 50 concurrent agents share a single Runtime + Checkpointer.
- 20-agent chaos test with random tool failures.
- Adapter mock tests use `httpx.MockTransport` — no API key required.

### Validated against
- BFCL V3 with three Chinese LLMs on SiliconFlow:
  GLM-5.1 90.7%, DeepSeek-V4-Flash 90.7% (excl. service errors), Kimi-K2.6 55.7%.

[Unreleased]: https://github.com/seleeleemail-lgtm/lingyia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/seleeleemail-lgtm/lingyia/releases/tag/v0.1.0
