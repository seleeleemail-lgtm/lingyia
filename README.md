# Lingyia

> **lingyia** _/ˈlɪŋ.jɑ/_ — from Chinese **灵芽** ("sentient sprout"): the moment intelligence puts down its first root.

**A production-grade Python agent runtime.** Thin core. Pluggable backends. Tested under real benchmarks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-92%20passing-brightgreen.svg)](#testing)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

---

## What is it?

Lingyia is a small, opinionated **agent runtime** for building reliable AI agents on top of LLMs. It splits cleanly into two layers:

- **`lingyia_core`** — a domain-agnostic loop (~600 lines). Owns _how_ an agent runs: model decisions, parallel tool calls, pause/resume, timeout, retry, durability, telemetry.
- **`lingyia_kit`** — production batteries. Provider adapters, generic tools, checkpointers, telemetry sinks, secret backends, PII redaction, circuit breakers, rate limiters, token compactors.

You compose them into a `Harness` that describes _what_ your agent does in a specific domain (contract review, code review, customer support, …). Swap harnesses to swap domains. The runtime never changes.

---

## Why another agent framework?

The 2024–2026 wave produced many frameworks. Lingyia takes a different bet:

- **Thin core, fat kit.** ~600 lines of orchestration in `lingyia_core`. Everything else is a swappable backend.
- **Production-grade out of the box.** Durable checkpoints, OTel telemetry, cost budgets, circuit breakers, rate limits, secret backends, PII redaction. Not a roadmap. Working code.
- **Zero hard dependencies beyond `httpx`.** Anthropic SDK, tiktoken, OpenTelemetry — all opt-in.
- **Validated on real benchmarks.** BFCL V3 runner ships in the repo. We measured GLM-5.1 at 90.7% accuracy with this runtime.
- **Multi-provider, OpenAI-compatible by default.** OpenAI, Anthropic, SiliconFlow, MiniMax, and any provider that speaks the OpenAI Chat Completions API.

---

## Install

```bash
pip install lingyia                  # core + httpx only
pip install "lingyia[anthropic]"     # + Anthropic Claude
pip install "lingyia[opentelemetry]" # + OTel tracing
pip install "lingyia[tiktoken]"      # + accurate token estimation
pip install "lingyia[all]"           # everything
```

Until v0.1.0 lands on PyPI, install from source:

```bash
pip install git+https://github.com/seleeleemail-lgtm/lingyia.git
```

---

## Hello, agent

```python
import asyncio
from lingyia_core import Runtime
from lingyia_kit import SiliconFlowModel, react_harness, read_file_tool

async def main():
    model = SiliconFlowModel(api_key="sk-...")
    harness = react_harness(tools=[read_file_tool(root="./docs")])
    runtime = Runtime.dev(model=model, max_iterations=6)

    result = await runtime.arun(
        harness,
        goal="Read README.md and summarize the public API in one paragraph.",
    )
    print(result.summary)

asyncio.run(main())
```

That's the dev path. The production path uses `Runtime.production(...)` which forces explicit durability + telemetry.

---

## Production setup

```python
from lingyia_core import Runtime
from lingyia_kit import SiliconFlowModel, react_harness
from lingyia_kit.checkpointers import SqliteCheckpointer
from lingyia_kit.telemetry import OTelTelemetrySink
from lingyia_kit.compactors import TokenAwareCompactor, tiktoken_estimator
from lingyia_kit.resilience import CircuitBreaker, TokenBucketRateLimiter, ProtectedModel
from lingyia_kit.secrets import EnvSecretBackend, DotenvSecretBackend, ChainSecretBackend
from lingyia_kit.pricing import estimate_cost

secrets = ChainSecretBackend(EnvSecretBackend(), DotenvSecretBackend(".env"))

protected_model = ProtectedModel(
    SiliconFlowModel(api_key=secrets.get("SILICONFLOW_API_KEY")),
    circuit_breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=30),
    rate_limiter=TokenBucketRateLimiter(rate_per_sec=10, burst=20),
)

runtime = Runtime.production(
    model=protected_model,
    checkpointer=SqliteCheckpointer("/var/lingyia/state.db"),
    telemetry=OTelTelemetrySink(),  # routes to Langfuse / Datadog / Jaeger via OTel
    compactor=TokenAwareCompactor(max_tokens=80_000, token_estimator=tiktoken_estimator("gpt-4o")),
    max_cost_usd=10.0,
    cost_estimator=lambda u: estimate_cost(
        u.model_id, u.prompt_tokens, u.completion_tokens, u.cached_tokens
    ),
)
```

---

## Features

| Capability | Status |
|---|---|
| **Async core loop** with parallel tool calls | ✅ |
| **Pause / resume** with unified `Interrupt` (approval + question) | ✅ |
| **Cross-process durability**: `SqliteCheckpointer` / `PostgresCheckpointer` / `RedisCheckpointer` | ✅ |
| **OpenTelemetry sink** + JSONL + structured logs | ✅ |
| **Token + cost tracking**, per-run budget, abort on overspend | ✅ |
| **Token-aware compaction** with pluggable estimator | ✅ |
| **Circuit breaker + rate limiter** for provider resilience | ✅ |
| **Secret backends** (env / dotenv / chain / Vault-ready) | ✅ |
| **Tool permission gating** with allow-list enforcement | ✅ |
| **PII redaction** for telemetry pipelines | ✅ |
| **Streaming run events** via `Runtime.astream` | ✅ |
| **Sub-agent / agent-as-tool** via `sub_agent_tool` | ✅ |
| **Sandboxed tools** (fs / http / shell with allow-lists) | ✅ |
| **Multi-provider adapters** (OpenAI / Anthropic / SiliconFlow / MiniMax) | ✅ |
| **BFCL V3 benchmark runner** for tool-calling accuracy | ✅ |
| **Real OTLP exporter wiring** for Langfuse / Datadog / Tempo / Jaeger | docs: [OTEL_SETUP.md](docs/OTEL_SETUP.md) |

---

## Architecture (one paragraph)

A `Runtime` orchestrates: model decision → guard check → tool execution → validator → repeat, with full instrumentation. A `Harness` injects domain content: `tools`, `validator`, `guard`, `context_builder`. Five `Protocol` types decouple the orchestrator from any concrete backend: `Model`, `Checkpointer`, `Compactor`, `TelemetrySink`, `ToolSchema`. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture.

```
            ┌─────────── lingyia_kit (L2) ───────────┐
            │  adapters  tools  checkpointers        │
            │  telemetry compactors resilience       │
            │  secrets   redaction patterns          │
            └────────────────────────────────────────┘
                                ▲ implements
                                │
            ┌─────────── lingyia_core (L1) ──────────┐
            │  Runtime  Harness  Tool  Decision      │
            │  Protocols: Model Checkpointer ...     │
            └────────────────────────────────────────┘
```

---

## Testing

```bash
pip install "lingyia[dev]"
pytest
```

Current coverage: **92 tests** including end-to-end integration (httpx mock + Sqlite checkpoint + JSONL telemetry + PII redaction + cost), 50 concurrent agents load test, and 20-agent chaos test with random tool failures.

---

## Benchmark results

Lingyia ships with a [BFCL V3](https://gorilla.cs.berkeley.edu/leaderboard.html) single-turn runner. We measured three Chinese LLMs via SiliconFlow:

| Model | Simple | Parallel | Multiple | Avg |
|---|---|---|---|---|
| **GLM-5.1** | 90% | 93% | 89% | **90.7%** |
| DeepSeek-V4-Flash (excl. service errors) | 94% | 90% | 88% | **90.7%** |
| Kimi-K2.6 (SiliconFlow function-calling has bugs) | 64% | 66% | 37% | 55.7% |

Run on your own provider:

```bash
SILICONFLOW_API_KEY=sk-... python -m lingyia_kit.eval.bfcl_runner \
  --models Pro/zai-org/GLM-5.1 \
  --splits simple,parallel,multiple --limit 100
```

---

## Status

**v0.1.0 — alpha.** The public API is settling but may change before v1. Use a pinned version in production.

We pin Python 3.9 as the minimum because some real-world environments are still there. Tested on 3.9 / 3.10 / 3.11 / 3.12.

---

## Roadmap (short)

- Inspect AI bridge — open the door to 200+ external benchmarks
- Reflection / replan pattern alongside ReAct
- Streaming token deltas (currently we stream events, not token-level chunks)
- A first-party `LangfuseSink` shortcut on top of the generic OTel sink
- More example projects: real legal review, code review, customer support

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). The project is small enough that a single PR can move the needle.

---

## License

[MIT](LICENSE) © 2026 Lingyia Contributors

---

## 中文简介

**Lingyia（灵芽）**是一个面向生产环境的 Python AI agent 运行时。设计哲学：**核心薄、扩展厚、零强制依赖**。

- `lingyia_core`：约 600 行领域无关的主循环
- `lingyia_kit`：生产级扩展（持久化、可观测、限流熔断、密钥管理、PII 脱敏、token 估算 …）

跟 LangChain / LangGraph 的差异：**我们不试图覆盖所有场景，只把"生产级 agent 必须做对的几件事"做对**。换领域 = 换 Harness，runtime 完全不动。

实测在 BFCL V3 function-calling 基准上，**GLM-5.1 + Lingyia 达到 90.7%**（接近公开 SOTA）。

欢迎贡献。中文 issue 在 [GitHub Discussions](https://github.com/seleeleemail-lgtm/lingyia/discussions) 也接受。
