# Architecture

This document explains the design ideas behind Lingyia. If you want to read code, start with `lingyia_core/runtime.py` (the loop) and `lingyia_core/protocols.py` (the contracts).

---

## 1. The split: Runtime vs Harness

The framework's central design choice is to put two different concerns in two different objects:

| | What it owns | Changes |
|---|---|---|
| **`Runtime`** | How an agent runs — model turn, parallel tools, guard, validator, interrupts, checkpoints, compaction, telemetry, timeouts, retries, cost budgets | Once per service deployment |
| **`Harness`** | What the agent does — its tools, its validator (when is the task done?), its guard (which actions need approval?), its context builder (what does it tell the model on every turn?), its permission grants | Once per domain / per agent type |

This is the "thin core, fat kit" thesis. **Lingyia bets that production concerns are the same across domains.** Legal contract review, code review, customer support, sales outreach — they all need durable checkpoints, retries, cost budgets, telemetry. None of them needs the runtime to know what a "contract" is.

A consequence: to ship a new domain agent, you write a new `Harness` and reuse the same `Runtime`. To swap LLM providers, you swap one adapter and the Harness doesn't change.

---

## 2. The six layers

```
┌──────────────────────────────────────────────────────────┐
│ L5  Product            Slack / Email / MCP / UI / Billing│   (your product)
├──────────────────────────────────────────────────────────┤
│ L4  Platform           Multi-tenancy / Quotas / Dashboard│   (your SaaS)
├──────────────────────────────────────────────────────────┤
│ L3  Application        LegalAgent / CodeReviewer / ...   │   (your code)
├──────────────────────────────────────────────────────────┤
│ L2  Composition        lingyia_kit                       │ ← we ship batteries
│     adapters / tools / memory / patterns / eval / ...    │
├──────────────────────────────────────────────────────────┤
│ L1  Runtime            lingyia_core                      │ ← we ship the loop
│     Runtime / Harness / Protocols / State / Executor     │
├──────────────────────────────────────────────────────────┤
│ L0  Provider           Anthropic / OpenAI / Google / ... │   (somebody else's)
└──────────────────────────────────────────────────────────┘
```

Lingyia ships L1 + L2. You bring L3+.

---

## 3. The five Protocols (the contracts)

`lingyia_core/protocols.py` defines five `typing.Protocol` types that let any concrete backend slot into the Runtime without code changes:

| Protocol | Implementations in `lingyia_kit` | Future / external |
|---|---|---|
| `Model` | OpenAI / Anthropic / SiliconFlow / MiniMax | Bedrock, Vertex AI, local llama.cpp |
| `Tool` (schema) | Generic tools (fs / http / shell) | Your domain tools |
| `Checkpointer` | `InMemory`, `SqliteCheckpointer` | `PostgresCheckpointer`, `RedisCheckpointer` |
| `Compactor` | `NoOpCompactor`, `TokenAwareCompactor` | LLM-based summarizer |
| `TelemetrySink` | `StructuredLog`, `JSONL`, `OTel` | Langfuse / Datadog / custom |

Implementing one of these is the only thing you need to extend Lingyia with a new backend. The Runtime never imports concrete classes from `lingyia_kit`.

---

## 4. The core loop (in pseudocode)

```python
while iteration < max_iterations:
    if compactor.should_compact(state):
        state = compactor.compact(state)

    decision = await model.adecide(context, state, tools)
    # decision carries optional ModelUsage -> cost tracking

    if decision.kind == FINAL_ANSWER:
        if validator(state).needs_human: pause(QUESTION)
        return COMPLETED

    if decision.kind == ASK_HUMAN:    pause(QUESTION)
    if decision.kind == ABORT:        return FAILED

    # CALL_TOOL
    if guard(decision).requires_approval: pause(APPROVAL, pending=decision)
    if not guard(decision).allowed:       feedback + continue

    for call in decision.tool_calls (in parallel):
        if missing permissions: ok=False, error="permission denied"
        result = executor.execute(tool, call)  # timeout + retry
        observations.append(result)

    verdict = validator(state)
    if verdict.done:        return COMPLETED
    if verdict.needs_human: pause(QUESTION)

    if max_cost_usd > 0 and state.cost_usd > max_cost_usd:
        return FAILED ("budget exceeded")
```

The full implementation is ~250 lines in `lingyia_core/runtime.py`. It does not know what a tool _does_; it only knows the contracts.

---

## 5. Pause / resume

Every pause in Lingyia is a single typed event: `Interrupt`. It carries:
- `reason` — `APPROVAL` (model wants to call a tool that needs human OK) or `QUESTION` (model asked the user something or validator escalated).
- `message` — what to show the operator.
- `pending_decision` — for `APPROVAL`, the exact tool call to run on resume.

Resume is just: load state, decide approved/rejected (or supply user answer), continue. The same code path handles "approve this risky action" and "answer this clarifying question".

State is JSON-serializable end-to-end (`RunState.to_dict()` / `RunState.from_dict()`), so resume can happen in a different process, hours later, after a service restart. We test this directly with `SqliteCheckpointer`.

---

## 6. Production runtime concerns (and where they live)

Following Codex's framing from the design review:

| Concern | Where it lives | Why |
|---|---|---|
| Timeouts | `Runtime` + `Tool.timeout_s` override | Loop must enforce |
| Retries | `RetryPolicy` injected into `Tool` | Per-tool semantics |
| Idempotency | `Tool.idempotent` + `side_effect` metadata | Needs durability layer to enforce (P1) |
| Checkpointing | `Checkpointer` Protocol; runtime calls it | Production-defining |
| Compaction | `Compactor` Protocol; runtime triggers | Long runs would crash without |
| Cost / budget | `Runtime.max_cost_usd` + `cost_estimator` | Cross-cutting |
| Telemetry | `TelemetrySink` Protocol; runtime emits | Cross-cutting |
| Rate limiting | `ProtectedModel` wrapper | Wrap-the-model pattern |
| Circuit breaking | `ProtectedModel` wrapper | Same |
| Secrets | `SecretBackend` Protocol; loaded by user code | Loaded once, injected as keys |
| Permission gating | `Tool.required_permissions` + `Harness.granted_permissions` | Runtime checks before exec |
| PII redaction | `redactor` arg on telemetry sinks | At the egress |
| Human interrupt | `Interrupt` typed object | Unified for approval + question |

---

## 7. What Lingyia deliberately does NOT do

| | Why |
|---|---|
| **Replace LangChain / LangGraph** | Different scope. They cover L1+L2+L3+more. We stop at L2. |
| **Build a "everything for everyone" framework** | Concrete domains are out of scope. You write your own L3. |
| **Hide the model API** | Adapters translate, they do not abstract away. You can drop down to provider-native code if you need to. |
| **Provide a vector DB / RAG layer** | That's L2 memory — when we add it, it will be a `MemoryBackend` Protocol with adapters, not a framework opinion. |
| **Multi-agent orchestration as core** | Agent-as-tool will land as a `SubAgentTool` pattern in `lingyia_kit`, not as a new top-level abstraction. The L1 loop already supports it conceptually. |

---

## 8. Versioning + stability

- `lingyia_core` public API will be **stable from v1.0**. We will add but not break.
- `lingyia_kit` may rev its internal modules more freely; the public adapter / sink / tool factory names are part of the API.
- Until v1.0 (alpha period), expect changes. Use a pinned version.

---

## 9. Where to read next

- `lingyia_core/runtime.py` — the loop
- `lingyia_core/protocols.py` — the five contracts
- `lingyia_core/state.py` — typed runtime state + serialization
- `lingyia_kit/adapters/_openai_base.py` — how an adapter implements `Model`
- `lingyia_kit/checkpointers/sqlite.py` — how a checkpointer implements `Checkpointer`
- `lingyia_kit/eval/bfcl_runner.py` — how we measure real performance

Open a [Discussion](https://github.com/seleeleemail-lgtm/lingyia/discussions) if you have architectural questions.
