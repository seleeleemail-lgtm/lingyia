## What this PR does

One short paragraph. Why this change matters more than what it does.

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behavior change)
- [ ] Docs
- [ ] Tests
- [ ] Build / CI / chore

## Affected layer

- [ ] `lingyia_core` (L1 runtime)
- [ ] `lingyia_kit` (L2 backends)
- [ ] Tests
- [ ] Docs / examples

## Checklist

- [ ] Tests pass locally (`pytest` or `python -m unittest`)
- [ ] mypy is happy on changed files (`mypy lingyia_core lingyia_kit`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Public API changes are documented in docstrings + README
- [ ] No new hard dependency added to `lingyia_core`
- [ ] New `lingyia_kit` extras (if any) are gated by `pyproject.toml`'s `optional-dependencies`

## Related issues

Closes #...
