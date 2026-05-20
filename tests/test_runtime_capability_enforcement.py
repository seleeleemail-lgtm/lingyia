"""Regression tests for Runtime capability enforcement (codex P2 §3, §4).

Spec §16 + spec §4.5/§4.7:
- Runtime MUST require Model.capabilities; missing or wrong-type fails hard
  with a real error (not silently skipped).
- Runtime MUST check decision.content emitted block kinds against
  capabilities.emits; reject anything outside.
"""
from __future__ import annotations

import asyncio

import pytest

from lingyia_core import (
    Decision,
    DecisionKind,
    Harness,
    Runtime,
    Tool,
    ToolResult,
)
from lingyia_core.blocks import (
    BlockKind,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    TruncationBlock,
)
from lingyia_core.capability import (
    CapabilityMismatchError,
    ModelCapabilities,
)
from lingyia_core.message import Message
from lingyia_core.runtime import CapabilityViolationError


_TEXT_ONLY_CAPS = ModelCapabilities(
    model_id="text-only",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
    emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
)


class _FinalAnswerModel:
    """Model that returns a scripted final-answer Decision."""

    capabilities = _TEXT_ONLY_CAPS

    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    async def adecide(self, ctx, state, tools):
        return self.decision


class _ModelMissingCapabilities:
    """Adapter without a capabilities attribute. Runtime MUST refuse to run."""

    async def adecide(self, ctx, state, tools):  # pragma: no cover — never reached
        return Decision.final_answer("hi")


class _ModelDictCapabilities:
    """Adapter whose capabilities is a wrong-shaped dict. MUST be rejected."""

    capabilities = {"model_id": "fake"}  # not a ModelCapabilities

    async def adecide(self, ctx, state, tools):  # pragma: no cover
        return Decision.final_answer("hi")


# ---------------------------------------------------------------------------
# capabilities presence (P2 §1)
# ---------------------------------------------------------------------------


def test_runtime_rejects_model_without_capabilities():
    """Hard fail: no `capabilities` attribute is a contract violation."""
    rt = Runtime.dev(model=_ModelMissingCapabilities())
    harness = Harness(tools=[])
    with pytest.raises(TypeError):
        asyncio.run(rt.arun(harness, "hi"))


def test_runtime_rejects_model_with_dict_capabilities():
    """Hard fail: capabilities must be a ModelCapabilities instance."""
    rt = Runtime.dev(model=_ModelDictCapabilities())
    harness = Harness(tools=[])
    with pytest.raises(TypeError):
        asyncio.run(rt.arun(harness, "hi"))


# ---------------------------------------------------------------------------
# emits enforcement (P2 §2)
# ---------------------------------------------------------------------------


def test_runtime_rejects_decision_with_block_outside_emits():
    """If a Model returns a block its capabilities.emits does NOT advertise,
    runtime raises CapabilityViolationError (spec §16 defensive check).
    """
    # ThinkingBlock is NOT in _TEXT_ONLY_CAPS.emits, but the model returns one.
    rogue = Decision.final_answer("hi")
    # Manually craft a decision with a ThinkingBlock to bypass Decision builders.
    rogue = Decision(
        kind=rogue.kind,
        content=(TextBlock(text="hi"), ThinkingBlock(thinking="leaked", signature="")),
    )
    rt = Runtime.dev(model=_FinalAnswerModel(rogue))
    harness = Harness(tools=[])
    with pytest.raises(CapabilityViolationError):
        asyncio.run(rt.arun(harness, "hi"))


def test_runtime_accepts_decision_within_emits():
    """Sanity: a decision whose blocks are subset of emits passes through."""
    decision = Decision.final_answer("ok")
    rt = Runtime.dev(model=_FinalAnswerModel(decision))
    harness = Harness(tools=[])
    result = asyncio.run(rt.arun(harness, "hi"))
    assert result.summary == "ok"


# ---------------------------------------------------------------------------
# runtime-authored block rejection (spec §6.1, codex P1.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_emitting_truncation_block_raises_before_enforce_emits():
    """Spec §6.1 (codex P1.1): _validate_no_runtime_blocks_from_model runs BEFORE
    _enforce_emits, so model-emitted TruncationBlock raises CapabilityViolationError
    (not ValueError from BlockKind('truncation'))."""

    class _RuntimeBlockEmittingModel:
        capabilities = ModelCapabilities(
            model_id="bad-model",
            accepts=frozenset({BlockKind.TEXT}),
            emits=frozenset({BlockKind.TEXT}),
        )

        async def adecide(self, ctx, state, tools):
            return Decision(
                kind=DecisionKind.FINAL_ANSWER,
                content=(TruncationBlock(count=5),),
            )

    runtime = Runtime.dev(_RuntimeBlockEmittingModel(), max_iterations=1)
    harness = Harness()

    with pytest.raises(CapabilityViolationError, match="TruncationBlock"):
        await runtime.arun(harness, "hello")
