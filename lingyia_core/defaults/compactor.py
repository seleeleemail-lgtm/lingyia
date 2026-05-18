"""Compactor defaults.

Production runtimes should provide a token-aware Compactor backed by the model's
tokenizer. This file provides the trivial NoOp default; real compactors live in
lingyia_kit (e.g. lingyia_kit.compactors.token_aware.TokenAwareCompactor for
v0.2 message-based compaction).
"""
from __future__ import annotations

from ..state import RunState


class NoOpCompactor:
    """Never compacts. Safe for short runs or when context overflow is acceptable."""

    def should_compact(self, state: RunState) -> bool:
        return False

    def compact(self, state: RunState) -> RunState:
        return state
