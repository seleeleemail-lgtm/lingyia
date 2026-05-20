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


# ---------------------------------------------------------------------------
# tool result validation (spec §6.2, codex P1.2 + P2.7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returning_truncation_block_in_output_raises():
    """Spec §6.2 (codex P1.2): tool returning (TruncationBlock,) as output raises
    CapabilityViolationError via _validate_tool_result after _serialize_tool_output."""

    async def bad_tool(args, ctx):
        return ToolResult(
            tool_name="bad_tool",
            output=(TruncationBlock(count=3),),
        )

    class _OneShotToolModel:
        capabilities = ModelCapabilities(
            model_id="m",
            accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
        )

        async def adecide(self, ctx, state, tools):
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                content=(ToolUseBlock(id="t1", name="bad_tool", input={}),),
            )

    tool = Tool.from_async(
        name="bad_tool",
        description="bad",
        handler=bad_tool,
    )
    runtime = Runtime.dev(_OneShotToolModel(), max_iterations=2)
    harness = Harness(tools=[tool])

    with pytest.raises(CapabilityViolationError, match="TruncationBlock"):
        await runtime.arun(harness, "go")


# ---------------------------------------------------------------------------
# aresume validation (spec §6.3, codex P2.8) — placed before nested test below
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aresume_rejects_truncation_block_in_pending_decision():
    """Spec §6.3 (codex P2.8): aresume() validates interrupt.pending_decision.content
    via shared helper before append/execute, closing checkpoint-injection vector."""
    from lingyia_core import RunState
    from lingyia_core.state import Interrupt, InterruptReason

    class _Model:
        capabilities = ModelCapabilities(
            model_id="m",
            accepts=frozenset({BlockKind.TEXT}),
            emits=frozenset({BlockKind.TEXT}),
        )

        async def adecide(self, ctx, state, tools):
            return Decision.final_answer("ok")

    # Simulate a corrupted checkpoint: RunState with poisoned pending_decision.
    poisoned_state = RunState(
        messages=[Message(role=Role.USER, content=(TextBlock(text="go"),))],
        run_id="r-poison",
        interrupt=Interrupt(
            reason=InterruptReason.APPROVAL,
            message="approve pending action",
            pending_decision=Decision(
                kind=DecisionKind.FINAL_ANSWER,
                content=(TruncationBlock(count=99),),  # POISONED
            ),
            iteration=1,
        ),
    )

    runtime = Runtime.dev(_Model(), max_iterations=2)
    harness = Harness()

    with pytest.raises(CapabilityViolationError, match="TruncationBlock"):
        await runtime.aresume(harness, poisoned_state, approved=True, feedback="")


@pytest.mark.asyncio
async def test_tool_returning_nested_truncation_block_raises():
    """Spec §6.2 (codex P2.7): _find_runtime_block recurses into nested
    ToolResultBlock.content. Tool returning ToolResultBlock(content=(TruncationBlock,))
    is caught by the recursive helper."""
    from lingyia_core.blocks import ToolResultBlock

    async def nested_bad_tool(args, ctx):
        return ToolResult(
            tool_name="nested_bad_tool",
            output=(
                ToolResultBlock(
                    tool_use_id="inner",
                    content=(TruncationBlock(count=7),),
                ),
            ),
        )

    class _OneShotToolModel:
        capabilities = ModelCapabilities(
            model_id="m",
            accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
        )

        async def adecide(self, ctx, state, tools):
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                content=(ToolUseBlock(id="t1", name="nested_bad_tool", input={}),),
            )

    tool = Tool.from_async(
        name="nested_bad_tool",
        description="bad",
        handler=nested_bad_tool,
    )
    runtime = Runtime.dev(_OneShotToolModel(), max_iterations=2)
    harness = Harness(tools=[tool])

    with pytest.raises(CapabilityViolationError, match="TruncationBlock"):
        await runtime.arun(harness, "go")


@pytest.mark.asyncio
async def test_tool_returning_truncation_block_in_list_typed_content_raises():
    """Codex post-impl P2.1: _find_runtime_block must recurse into list-typed
    ToolResultBlock.content (Union[str, tuple] is not enforced at runtime;
    list could slip through and bypass §6.2)."""
    from lingyia_core.blocks import ToolResultBlock

    async def list_typed_bad_tool(args, ctx):
        # content is a LIST, not a tuple — type hint not enforced at runtime
        return ToolResult(
            tool_name="list_typed_bad_tool",
            output=(
                ToolResultBlock(
                    tool_use_id="inner",
                    content=[TruncationBlock(count=11)],  # LIST!
                ),
            ),
        )

    class _OneShotToolModel:
        capabilities = ModelCapabilities(
            model_id="m",
            accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
        )

        async def adecide(self, ctx, state, tools):
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                content=(ToolUseBlock(id="t1", name="list_typed_bad_tool", input={}),),
            )

    tool = Tool.from_async(
        name="list_typed_bad_tool",
        description="bad",
        handler=list_typed_bad_tool,
    )
    runtime = Runtime.dev(_OneShotToolModel(), max_iterations=2)
    harness = Harness(tools=[tool])

    with pytest.raises(CapabilityViolationError, match="TruncationBlock"):
        await runtime.arun(harness, "go")
