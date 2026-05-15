# Contributing to Lingyia

Thanks for your interest. The project is small and the bar for "good change" is high. Here is what we look for and how to ship it.

## Ground rules

- **Tests stay green.** Every PR runs all 92 tests across Python 3.9–3.12 via CI. Red CI is a hard block.
- **No new hard dependencies in `lingyia_core`.** It depends on Python stdlib only (plus the project's own modules). New backends belong in `lingyia_kit`.
- **No new hard dependencies in `lingyia_kit` without discussion.** `httpx` is the only required wire-level dep. Anthropic SDK, OpenTelemetry, tiktoken are all lazy-imported optional extras. Adding another opt-in extra is fine; adding a hard one needs an issue first.
- **Production concerns belong in `Runtime` or as Protocols.** Domain concerns belong in `Harness` / user code. If you find yourself adding a domain-specific field to `Runtime`, stop and discuss.

## Setup

```bash
git clone https://github.com/seleeleemail-lgtm/lingyia.git
cd lingyia
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
pytest                 # 92 tests should pass
mypy lingyia_core lingyia_kit
```

## Workflow

1. **Open an issue first** for non-trivial changes. We respond to design proposals before merging implementation.
2. Branch from `main`. Use `kebab-case` branch names. Example: `feat/postgres-checkpointer`, `fix/streaming-cancellation`.
3. Keep PRs small. < 400 lines of diff is ideal. Refactors live in separate PRs from features.
4. Write tests **before** the implementation when possible. The repo's test culture is uncompromising — we measured BFCL on real APIs, we test 50 concurrent agents, we test chaos with random tool failures.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Update relevant docs (`README.md`, `docs/ARCHITECTURE.md`) if you change public API.

## Commit messages

Convention: `<type>(<scope>): <subject>` where type is `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Examples:

- `feat(checkpointer): PostgresCheckpointer with asyncpg`
- `fix(streaming): propagate CancelledError on consumer disconnect`
- `refactor(runtime): extract budget enforcement to dedicated method`

Multi-line body is encouraged for non-trivial changes. State **why**, not just **what**.

## Code style

- Type hints on every public function.
- Docstrings on every public class and public method. One paragraph minimum on why this thing exists, not just what it does.
- No unprefixed `print()` in library code; use structured telemetry instead.
- f-strings preferred. No `%`-formatting in new code.
- Single quotes for strings, double quotes for docstrings — pick one per file and stay consistent (we use double for both, but we won't reject a PR over this).

## Tests

- Unit tests in `lingyia_kit/tests/`. Core tests in `tests/`.
- Integration tests use `httpx.MockTransport` so they run offline.
- Load + chaos tests run as part of the regular suite — they take ~2 seconds.
- For LLM-dependent behaviors, prefer recording an API response and replaying it via MockTransport over hitting a live API in CI.

## What we will probably reject

- "Convenience" wrappers that hide the underlying API surface for no clear gain.
- New Pydantic v1 dependencies. Pydantic v2 is fine for opt-in features.
- "Compatibility shims" for frameworks we don't depend on (LangChain, etc.). If a user needs that bridge, it can live in their own package.
- PRs that loosen type checks to silence mypy.
- Performance optimizations without a benchmark showing the gain.

## What we are eager to merge

- Real provider adapters (Google Gemini, local llama.cpp, AWS Bedrock).
- Real benchmark bridges (Inspect, τ-bench, GAIA).
- Production telemetry exporters with worked examples (Langfuse, Datadog, Honeycomb).
- New Checkpointer backends (Postgres, Redis, S3).
- Real-world example projects in `examples/` showing a complete domain agent.

## Questions

Open a [Discussion](https://github.com/seleeleemail-lgtm/lingyia/discussions). 中文也欢迎。

By contributing you agree your work is licensed under the [MIT License](LICENSE).
