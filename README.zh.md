# Lingyia 灵芽

> **lingyia** _/ˈlɪŋ.jɑ/_ — 取自中文「**灵芽**」(sentient sprout)：智能落地生根的那一刻。

**面向生产环境的 Python AI agent 运行时。** 核心薄、扩展厚、真实 benchmark 验证。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-92%20passing-brightgreen.svg)](#测试)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#状态)

[English](README.md) | **中文**

---

## 它是什么？

Lingyia 是一个**精简、有立场**的 agent 运行时，用来在 LLM 之上构建可靠的 AI agent。两层架构：

- **`lingyia_core`** —— 领域无关的主循环（约 600 行）。负责 *agent 怎么跑*：模型决策、并行工具调用、暂停/恢复、超时、重试、持久化、可观测。
- **`lingyia_kit`** —— 生产组件电池。Provider 适配器、通用工具、checkpointer、telemetry sink、密钥后端、PII 脱敏、熔断器、限流器、token 压缩器。

你用 `Harness` 把它们组合起来，描述 agent 在**特定领域**做什么（合同审查、代码 review、客服…）。**换 harness 就换领域，runtime 永远不变**。

---

## 为什么再造一个 agent 框架？

2024-2026 年涌现了大量 agent 框架。Lingyia 押注一条不一样的路：

- **核心薄、扩展厚**。`lingyia_core` 约 600 行编排逻辑。其他都是可替换后端。
- **生产级开箱即用**。可持久化 checkpoint、OTel 可观测、成本预算、熔断器、限流、密钥管理、PII 脱敏。**不是 roadmap，是已经在跑的代码**。
- **零强制依赖**（除 `httpx`）。Anthropic SDK、tiktoken、OpenTelemetry 全是可选 extra。
- **真实 benchmark 验证**。仓库内置 BFCL V3 runner。我们实测 **GLM-5.1 准确率 90.7%**（接近 SOTA）。
- **多 provider、默认 OpenAI 兼容**。OpenAI / Anthropic / 硅基流动 / MiniMax，以及任何走 OpenAI Chat Completions 协议的服务。

---

## 安装

```bash
pip install lingyia                     # 仅 core + httpx
pip install "lingyia[anthropic]"        # + Anthropic Claude
pip install "lingyia[opentelemetry]"    # + OTel 链路追踪
pip install "lingyia[tiktoken]"         # + 精确 token 估算
pip install "lingyia[postgres]"         # + Postgres checkpointer
pip install "lingyia[redis]"            # + Redis checkpointer
pip install "lingyia[all]"              # 一次性全装
```

v0.1.0 还没上 PyPI 之前，从源码装：

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
        goal="读取 README.md 并用一段话总结公开 API。",
    )
    print(result.summary)

asyncio.run(main())
```

这是开发路径。生产路径用 `Runtime.production(...)`，强制要求显式的持久化 + telemetry。

---

## 生产配置

```python
from lingyia_core import Runtime
from lingyia_kit import SiliconFlowModel, react_harness
from lingyia_kit.checkpointers import PostgresCheckpointer
from lingyia_kit.telemetry import OTelTelemetrySink
from lingyia_kit.compactors import TokenAwareCompactor, tiktoken_estimator
from lingyia_kit.resilience import CircuitBreaker, TokenBucketRateLimiter, ProtectedModel
from lingyia_kit.secrets import EnvSecretBackend, DotenvSecretBackend, ChainSecretBackend
from lingyia_kit.pricing import estimate_cost

secrets = ChainSecretBackend(EnvSecretBackend(), DotenvSecretBackend(".env"))

# Model 套上熔断 + 限流
protected_model = ProtectedModel(
    SiliconFlowModel(api_key=secrets.get("SILICONFLOW_API_KEY")),
    circuit_breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=30),
    rate_limiter=TokenBucketRateLimiter(rate_per_sec=10, burst=20),
)

# 生产 Runtime（fail-closed：少了 checkpointer / telemetry 直接报错）
runtime = Runtime.production(
    model=protected_model,
    checkpointer=PostgresCheckpointer(dsn="postgres://user:pass@db/lingyia"),
    telemetry=OTelTelemetrySink(),  # 经 OTel 自动桥接到 Langfuse / Datadog / Jaeger
    compactor=TokenAwareCompactor(max_tokens=80_000, token_estimator=tiktoken_estimator("gpt-4o")),
    max_cost_usd=10.0,                # 单 run 预算上限
    cost_estimator=lambda u: estimate_cost(
        u.model_id, u.prompt_tokens, u.completion_tokens, u.cached_tokens
    ),
)
```

---

## 能力清单

| 能力 | 状态 |
|---|---|
| **异步主循环** + 并行 tool 调用 | ✅ |
| **暂停 / 恢复**，统一 `Interrupt`（审批 + 问询）| ✅ |
| **跨进程持久化**：`SqliteCheckpointer` / `PostgresCheckpointer` / `RedisCheckpointer` | ✅ |
| **OpenTelemetry sink** + JSONL + 结构化日志 | ✅ |
| **Token + cost 追踪**，per-run 预算，超支即 abort | ✅ |
| **Token-aware 上下文压缩**，可插拔 estimator | ✅ |
| **熔断 + 限流** 抵御 provider 抖动 | ✅ |
| **密钥后端**（env / dotenv / chain，预留 Vault）| ✅ |
| **工具权限网关** 强制白名单 | ✅ |
| **PII 脱敏** 用于 telemetry 流水线 | ✅ |
| **Run 事件流**：`Runtime.astream` | ✅ |
| **Sub-agent / agent-as-tool**：用 `sub_agent_tool` 递归调用 | ✅ |
| **沙箱工具**（fs / http / shell + allow-list）| ✅ |
| **多 provider adapter**（OpenAI / Anthropic / SiliconFlow / MiniMax）| ✅ |
| **BFCL V3 benchmark runner** | ✅ |

---

## 架构（一段话讲完）

`Runtime` 编排：模型决策 → guard 检查 → 工具执行 → validator → 循环，全程有埋点。`Harness` 注入领域内容：`tools`、`validator`、`guard`、`context_builder`。五个 `Protocol` 把编排器跟具体后端解耦：`Model`、`Checkpointer`、`Compactor`、`TelemetrySink`、`ToolSchema`。完整设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

```
            ┌─────── lingyia_kit (L2) ───────────────┐
            │  adapters  tools  checkpointers        │
            │  telemetry compactors resilience       │
            │  secrets   redaction patterns          │
            └────────────────────────────────────────┘
                                ▲ implements
                                │
            ┌─────── lingyia_core (L1) ──────────────┐
            │  Runtime  Harness  Tool  Decision      │
            │  Protocols: Model Checkpointer ...     │
            └────────────────────────────────────────┘
```

---

## 测试

```bash
pip install "lingyia[dev]"
pytest
```

当前覆盖：**92 个测试**，含端到端集成（httpx mock + sqlite 持久化 + JSONL telemetry + PII 脱敏 + 成本累计）、50 并发 agent 负载测试、20 agent + 随机 tool 失败的混沌测试。

---

## Benchmark 实测

Lingyia 内置 [BFCL V3](https://gorilla.cs.berkeley.edu/leaderboard.html) 单轮 runner。我们通过 SiliconFlow 测了三个国产 LLM：

| 模型 | Simple | Parallel | Multiple | 平均 |
|---|---|---|---|---|
| **GLM-5.1** | 90% | 93% | 89% | **90.7%** |
| DeepSeek-V4-Flash（排除服务错误）| 94% | 90% | 88% | **90.7%** |
| Kimi-K2.6（SiliconFlow function-calling 有 bug）| 64% | 66% | 37% | 55.7% |

跑你自己的 provider：

```bash
SILICONFLOW_API_KEY=sk-... python -m lingyia_kit.eval.bfcl_runner \
  --models Pro/zai-org/GLM-5.1 \
  --splits simple,parallel,multiple --limit 100
```

---

## 状态

**v0.1.0 — alpha**。公开 API 正在稳定，v1.0 之前可能变化。生产请固定版本号。

最低 Python 3.9（部分企业环境还在用）。测试覆盖 3.9 / 3.10 / 3.11 / 3.12。

---

## Roadmap（近期）

- Inspect AI bridge —— 一次接入 200+ 外部 benchmark
- Reflection / replan 模式（除 ReAct 之外）
- Streaming token deltas（目前流的是 event，不是 token 增量）
- 真实 OTLP exporter 完整示例（[`docs/OTEL_SETUP.md`](docs/OTEL_SETUP.md) 已起步）

---

## 贡献

欢迎 issue 和 PR，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。项目体量足够小，**一个 PR 就能产生显著影响**。

中文 issue 在 [Discussions](https://github.com/seleeleemail-lgtm/lingyia/discussions) 也接受。

---

## License

[MIT](LICENSE) © 2026 Lingyia Contributors

---

## 设计取舍（如果你想懂得更深）

**为什么不做 LangChain / LangGraph 的对手？**

- 范围不同。它们覆盖 L1+L2+L3+周边。我们停在 L2。
- 我们的判断：**生产 agent 需要做对的事情就那么几件，把它们做对，比覆盖 500 个 chain pattern 更有价值**。

**为什么 `lingyia_core` 没有 sub-agent / multi-agent 一等公民？**

- 因为不需要。Sub-agent 通过 `sub_agent_tool` 把另一个 Runtime 包装成 Tool 就可以了。**不需要新抽象**。这是 thin core 原则的实证。

**为什么开源？**

- 这个框架本身**不是商业护城河**（参考 LangChain → LangSmith / HashiCorp → Cloud 模式）。
- 真正的护城河是基于 Lingyia 做的**领域产品**（合同审查、代码 review、客服…）+ 客户数据 + 行业 know-how。
- 开源底座 = 技术 brand + 招聘工具 + 复利效应（让你能并行做多个领域 agent）。

---

如果你正在做 agent 产品，欢迎 [开 Discussion](https://github.com/seleeleemail-lgtm/lingyia/discussions) 跟我们聊。
