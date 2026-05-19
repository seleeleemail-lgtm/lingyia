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


def test_final_answer_rejected_no_feedback_continues_loop():
    """Codex verify P2 (runtime.py:506): a validator that rejects a FINAL_ANSWER
    with ``done=False`` but EMPTY feedback must still re-enter the loop — not
    silently complete with the rejected answer.

    The original fix only handled the compound ``not done AND feedback`` path.
    With ``feedback=""`` the runtime fell through to ``RunResult(COMPLETED,
    summary=decision.text)`` — i.e. it returned the rejected answer as the
    final result. This test pins the corrected behavior: keep looping until
    the validator approves or max_iterations is hit.
    """
    # Three rejected attempts followed by a valid one. With empty feedback
    # the runtime must still consume iterations and re-call the model.
    rejected_a = Decision.final_answer("attempt A")
    rejected_b = Decision.final_answer("attempt B")
    rejected_c = Decision.final_answer("attempt C")
    accepted = Decision.final_answer("attempt D")

    model = _ScriptedModel(rejected_a, rejected_b, rejected_c, accepted)

    def empty_feedback_validator(state):
        # Look at most recent assistant text
        last_answer = None
        for msg in reversed(state.messages):
            if msg.role == Role.ASSISTANT:
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        last_answer = b.text
                        break
                if last_answer is not None:
                    break
        if last_answer == "attempt D":
            return ValidationResult(done=True, summary="finally ok")
        # Reject without feedback — runtime must still loop, not complete.
        return ValidationResult(done=False, feedback="")

    harness = Harness(validator=empty_feedback_validator)
    rt = Runtime.dev(model=model, max_iterations=8)
    result = asyncio.run(rt.arun(harness, "go"))

    assert result.status == RunStatus.COMPLETED, (
        f"runtime completed with rejected answer: status={result.status} "
        f"summary={result.summary!r} reason={result.reason!r}"
    )
    assert result.summary == "finally ok"
    assert model.calls == 4

    # No USER TextBlock should mention "attempt" because feedback was empty.
    # (Sanity that empty-feedback path didn't accidentally inject a marker.)
    user_text_messages = [
        b.text
        for m in result.state.messages if m.role == Role.USER
        for b in m.content if isinstance(b, TextBlock)
    ]
    # Only the original "go" should be present.
    assert user_text_messages == ["go"], user_text_messages


def test_final_answer_rejected_no_feedback_hits_max_iterations():
    """If the validator keeps rejecting without feedback and the model keeps
    returning the same final answer, the run must terminate at
    max_iterations — not loop forever, and not silently complete.
    """
    rejected = Decision.final_answer("nope")
    # Provide many; the runtime should stop at max_iterations regardless.
    model = _ScriptedModel(*([rejected] * 10))

    harness = Harness(validator=lambda s: ValidationResult(done=False, feedback=""))
    rt = Runtime.dev(model=model, max_iterations=3)
    result = asyncio.run(rt.arun(harness, "go"))

    # Must NOT complete with the rejected answer.
    assert result.status != RunStatus.COMPLETED, (
        f"runtime falsely completed with rejected answer: summary={result.summary!r}"
    )
    # Should have stopped at max_iterations (current runtime returns STOPPED
    # with reason 'max_iterations reached').
    assert result.status == RunStatus.STOPPED
    assert "max_iterations" in (result.reason or "")
    assert model.calls == 3
