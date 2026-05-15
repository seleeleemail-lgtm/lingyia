"""Token-aware Compactor.

Drops oldest observations when the estimated token cost of the run's context
exceeds a configured budget. Leaves a single ``compaction_marker`` observation
in their place so the model knows context was elided rather than silently lost.

By default uses a cheap ``len(text) // 4`` estimator. Pass ``tiktoken_estimator``
(or any callable ``(str) -> int``) for accurate per-model token counts.
"""
from __future__ import annotations

import copy
import json
from typing import Callable, Optional

from lingyia_core import RunState
from lingyia_core.state import Observation


TokenEstimator = Callable[[str], int]


def char_div4_estimator(text: str) -> int:
    """Crude default: 1 token ≈ 4 chars. Good enough as a fallback."""
    return max(1, len(text) // 4)


def tiktoken_estimator(model: str = "gpt-4o") -> TokenEstimator:
    """Build a tiktoken-backed estimator. Falls back to char_div4 if the
    ``tiktoken`` package is not installed or the model is unknown."""
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return char_div4_estimator
    try:
        encoder = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return char_div4_estimator

    def _est(text: str) -> int:
        if not text:
            return 0
        return len(encoder.encode(text))

    return _est


class TokenAwareCompactor:
    """Compactor that enforces a token budget.

    Strategy:
    - Estimate total token cost of goal + observations + feedback.
    - If over budget, keep the last ``keep_last_observations`` observations
      and prepend a single ``compaction_marker`` observation summarizing the
      drop.
    - Also truncates feedback to ``keep_recent_feedback`` entries.

    The estimator is pluggable so production users can wire in a real
    tokenizer (tiktoken, transformers) without changing this code.
    """

    def __init__(
        self,
        max_tokens: int = 80_000,
        keep_last_observations: int = 10,
        keep_recent_feedback: int = 20,
        token_estimator: Optional[TokenEstimator] = None,
        summary_marker: str = "[earlier context truncated]",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.keep_last = keep_last_observations
        self.keep_feedback = keep_recent_feedback
        self.estimator: TokenEstimator = token_estimator or char_div4_estimator
        self.marker = summary_marker

    def _estimate_state_tokens(self, state: RunState) -> int:
        total = self.estimator(state.goal or "")
        for obs in state.observations:
            try:
                payload_text = json.dumps(obs.payload, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                payload_text = str(obs.payload)
            total += self.estimator(payload_text)
        for fb in state.feedback:
            total += self.estimator(fb)
        return total

    def should_compact(self, state: RunState) -> bool:
        return self._estimate_state_tokens(state) > self.max_tokens

    def compact(self, state: RunState) -> RunState:
        new = copy.copy(state)
        # observations and feedback are mutable lists on the dataclass; copy
        # them so the original RunState isn't mutated.
        new.observations = list(state.observations)
        new.feedback = list(state.feedback)
        new.trace = list(state.trace)
        new.metadata = dict(state.metadata)

        if len(new.observations) > self.keep_last:
            dropped = len(new.observations) - self.keep_last
            kept = new.observations[-self.keep_last:]
            first_iter = kept[0].iteration if kept else 0
            marker = Observation(
                iteration=first_iter,
                kind="compaction_marker",
                payload={
                    "summary": self.marker,
                    "observations_dropped": dropped,
                },
            )
            new.observations = [marker] + kept

        if len(new.feedback) > self.keep_feedback:
            new.feedback = list(state.feedback[-self.keep_feedback:])

        return new
