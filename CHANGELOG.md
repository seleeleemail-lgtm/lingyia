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

## [0.2.0-alpha] — 2026-05-18 — Message Contract Reset

This is a **breaking** release. The internal state contract was rebuilt from
the ground up around a message log (Anthropic-style content blocks) instead of
the v0.1 `goal + observations + feedback` triple. v0.1 snapshots are not
loadable directly — see Migration below.

### BREAKING CHANGES
- `RunState`: removed `goal: str`, `observations: list[Observation]`,
  `feedback: list[str]`. Added `messages: list[Message]` as the sole source
  of truth for run history. `schema_version` bumped 1 → 2. Loading a v0.1
  snapshot via `RunState.from_dict` raises `UnknownSchemaVersionError`.
- `Decision.content`: changed from `str` to `tuple[ContentBlock, ...]`. Use
  `decision.text` for the flat string and `decision.tool_calls` for the
  derived `ToolUseBlock` tuple.
- `Model` protocol: now requires a `capabilities: ModelCapabilities`
  property. Custom Model implementations must declare what they accept/emit.
- `ToolContext`: removed `goal: str`, added `messages: tuple[Message, ...]`.
  Tools that read goal text must now read it off the first user message.

### Added
- `lingyia_core.blocks`: `ContentBlock` discriminated union covering
  `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`, `AudioBlock`,
  and `ThinkingBlock`. `block_to_dict` / `block_from_dict` round-trip
  helpers, plus `Role` and `BlockKind` enums.
- `lingyia_core.message.Message`: dataclass with `role: Role` and
  `content: tuple[ContentBlock, ...]`. Frozen, JSON-serializable.
- `lingyia_core.capability`: `ModelCapabilities` dataclass (accepted/emitted
  `BlockKind` sets, `max_context_tokens`, etc.), `CapabilityMismatchError`,
  `CapabilityPolicy` protocol, and `FailFastCapabilityPolicy` default.
  Runtime checks model capabilities against the planned decision before
  each `adecide` call.
- `Harness.system_prompt` and `Harness.metadata` fields. The system prompt
  is auto-injected as a `SYSTEM` message at run start.
- AnthropicModel: rewritten with native `ContentBlock` ↔ Anthropic API block
  mapping (still raw `httpx`, no `anthropic` SDK dependency). Supports
  `ThinkingBlock` round-trip.
- Per-model capabilities on OpenAI / SiliconFlow / MiniMax / Anthropic
  adapters with real context windows (GLM-5.1 = 205K, Kimi-K2.6 = 200K,
  DeepSeek-V4 = 128K, MiniMax-M2 = 200K, Claude family, gpt-4o, etc.).
- `lingyia_kit.migrations.v01_to_v02.migrate_state_dict`: best-effort
  v0.1 → v0.2 snapshot dict conversion. Lossy on observation timestamps
  and the exact interleaving of `feedback` strings — see the docstring for
  caveats.
- `lingyia_kit.compactors.token_aware.TokenAwareCompactor`: rewritten for
  messages — preserves the `SYSTEM` message, keeps the last N turn pairs,
  and enforces `tool_use` ↔ `tool_result` adjacency so the conversation
  never desyncs after compaction.
- `lingyia_core` now re-exports `Message`, `Role`, `BlockKind`,
  `ContentBlock`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`,
  `ImageBlock`, `AudioBlock`, `ThinkingBlock`, `ModelCapabilities`,
  `CapabilityMismatchError`, `CapabilityPolicy`,
  `FailFastCapabilityPolicy`, and `UnknownSchemaVersionError` at the
  package root.

### Changed
- `Runtime`: full message-lifecycle implementation. Appends an `ASSISTANT`
  message after every `adecide`; appends a `USER` message containing a
  `ToolResultBlock` after every tool execution; appends a `USER` message
  with a `TextBlock` for guard / validator / approval-reject /
  human-resume feedback. Capability check runs before `adecide` via the
  configured `CapabilityPolicy`.
- `Runtime.arun` / `Runtime.astream` now accept either a `str` (auto-wrapped
  to a single `USER` message) or a pre-built `list[Message]`. The legacy
  `goal=` keyword is still accepted as a backwards-compat shim.
- `lingyia_kit.patterns.react`: simplified — no goal/observations context
  builder. Just `tools + system_prompt + metadata + permissions`.

### Removed
- `TruncatingCompactor` (it operated on v0.1 `observations`/`feedback`).
- v0.1 `RunState` fields: `goal`, `observations`, `feedback`.
- `RedactedBlock` was kept out of the `ContentBlock` union — redaction now
  lives in the telemetry sink layer where it belongs.
- AnthropicModel's v0.1 "reconstruct multi-turn assistant tool_calls from
  observations" reassembly logic. No longer needed; messages are the
  source of truth.

### Deprecated
- `Observation` dataclass: still exported from `lingyia_core` for
  type-import compatibility but is no longer used by `Runtime` or
  `RunState`. Slated for removal in v0.3.

### Migration
For callers persisting v0.1 snapshots:

```python
from lingyia_kit.migrations.v01_to_v02 import migrate_state_dict
from lingyia_core import RunState

v02_dict = migrate_state_dict(v01_snapshot_dict)
state = RunState.from_dict(v02_dict)
```

For programmatic v0.1 callers, construct a `RunState` directly:

```python
from lingyia_core import RunState, Message, Role, TextBlock

state = RunState(
    messages=[Message(role=Role.USER, content=(TextBlock(text=goal),))],
)
```

The `Runtime.arun(harness, goal="...")` shim still works for the common
"start a run from a string goal" case.

### Testing
138/138 tests green across `lingyia_core` and `lingyia_kit` on the v0.2-α
branch. All adapters (OpenAI / SiliconFlow / MiniMax / Anthropic),
checkpointers, compactors, resilience, telemetry, tools, and streaming
suites migrated to the new contract.

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
