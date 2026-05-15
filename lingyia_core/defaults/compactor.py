"""Compactor defaults.

Production runtimes should provide a token-aware Compactor backed by the model's
tokenizer. These defaults are deliberately simple: NoOp for short runs, and a
truncating compactor for length-bounded scenarios.
"""
from __future__ import annotations

import copy

from ..state import RunState


class NoOpCompactor:
    """Never compacts. Safe for short runs or when context overflow is acceptable."""

    def should_compact(self, state: RunState) -> bool:
        return False

    def compact(self, state: RunState) -> RunState:
        return state


class TruncatingCompactor:
    """Keeps the last N observations + feedback entries. Naive but predictable.

    For real workloads, replace with a token-counting summarizer.
    """

    def __init__(self, max_observations: int = 50, max_feedback: int = 20) -> None:
        self._max_obs = max_observations
        self._max_fb = max_feedback

    def should_compact(self, state: RunState) -> bool:
        return (
            len(state.observations) > self._max_obs
            or len(state.feedback) > self._max_fb
        )

    def compact(self, state: RunState) -> RunState:
        new = copy.copy(state)
        new.observations = list(state.observations[-self._max_obs:])
        new.feedback = list(state.feedback[-self._max_fb:])
        return new
