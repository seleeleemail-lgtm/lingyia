"""Regression tests for validator feedback lifecycle (codex P2 runtime.py:461).

The validator runs after both tool-execution turns and final-answer turns.
Per the validator lifecycle contract, a non-empty ``ValidationResult.feedback``
on a not-done verdict MUST be re-entered into the conversation as a USER
TextBlock and the loop must continue — same as the tool path
(_maybe_finish_after_tools).

Before the fix, FINAL_ANSWER ignored feedback unless ``needs_human`` was
true, silently swallowing the validator's rejection.
"""
from __future__ import annotations

import asyncio

from lingyia_core import (
    Decision,
    Harness,
    Runtime,
    RunStatus,
    ValidationResult,
)
from lingyia_core.blocks import BlockKind, Role, TextBlock
from lingyia_core.capability import ModelCapabilities


_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT}),
    emits=frozenset({BlockKind.TEXT}),
)


class _ScriptedModel:
    """Scripted model that returns a list of Decisions in order."""

    capabilities = _CAPS

    def __init__(self, *decisions: Decision) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    async def adecide(self, ctx, state, tools):
        self.calls += 1
        if not self._decisions:
            raise AssertionError("scripted model exhausted")
        return self._decisions.pop(0)


def test_validator_feedback_after_final_answer_reenters_loop():
    """A validator that rejects the final answer must be able to push the
    loop forward with feedback, just like the tool-completion path does.
    """
    rejected = Decision.final_answer("first attempt")
    accepted = Decision.final_answer("second attempt")

    model = _ScriptedModel(rejected, accepted)

    rejection_count = {"n": 0}

    def picky_validator(state):
        # Find the most recent assistant TextBlock — that's the answer
        # under consideration.
        last_answer = None
        for msg in reversed(state.messages):
            if msg.role == Role.ASSISTANT:
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        last_answer = b.text
                        break
                if last_answer is not None:
                    break
        if last_answer == "first attempt":
            rejection_count["n"] += 1
            return ValidationResult(
                done=False,
                feedback="not good enough; please try again",
            )
        if last_answer == "second attempt":
            return ValidationResult(done=True, summary="ok")
        return ValidationResult(done=False)

    harness = Harness(validator=picky_validator)
    rt = Runtime.dev(model=model, max_iterations=4)
    result = asyncio.run(rt.arun(harness, "go"))

    assert result.status == RunStatus.COMPLETED, result.reason
    assert result.summary == "ok"
    assert rejection_count["n"] == 1
    assert model.calls == 2

    # Last user-role message must be the feedback we injected.
    user_msgs = [m for m in result.state.messages if m.role == Role.USER]
    # Should be: original "go" then the feedback after rejection
    assert any(
        any(isinstance(b, TextBlock) and "not good enough" in b.text for b in m.content)
        for m in user_msgs
    ), "validator feedback was not pushed into the transcript"


def test_validator_with_no_feedback_still_completes():
    """Sanity: a done verdict (no feedback) finalizes immediately."""
    model = _ScriptedModel(Decision.final_answer("done"))
    harness = Harness(validator=lambda s: ValidationResult(done=True, summary="ok"))
    rt = Runtime.dev(model=model)
    result = asyncio.run(rt.arun(harness, "go"))
    assert result.status == RunStatus.COMPLETED
    assert result.summary == "ok"
    assert model.calls == 1
