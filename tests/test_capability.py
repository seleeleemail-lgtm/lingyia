"""Tests for ModelCapabilities + CapabilityPolicy (v0.2-α §4.5, §4.7)."""
import pytest

from lingyia_core.blocks import BlockKind, Role, TextBlock, ImageBlock, ImageSource
from lingyia_core.message import Message
from lingyia_core.capability import (
    ModelCapabilities, CapabilityMismatchError,
    CapabilityPolicy, FailFastCapabilityPolicy,
)


def test_model_capabilities_defaults():
    caps = ModelCapabilities(model_id="test")
    assert caps.accepts == frozenset({BlockKind.TEXT})
    assert caps.emits == frozenset({BlockKind.TEXT})
    assert caps.supports_streaming is True
    assert caps.supports_parallel_tools is True
    assert caps.supports_json_schema is False
    assert caps.max_context_tokens == 8192


def test_model_capabilities_custom():
    caps = ModelCapabilities(
        model_id="glm-5.1",
        accepts=frozenset({BlockKind.TEXT, BlockKind.IMAGE}),
        max_context_tokens=205_000,
    )
    assert caps.model_id == "glm-5.1"
    assert BlockKind.IMAGE in caps.accepts


def test_capability_mismatch_error_message():
    err = CapabilityMismatchError(
        required=frozenset({BlockKind.TEXT, BlockKind.IMAGE}),
        accepted=frozenset({BlockKind.TEXT}),
        unsupported=frozenset({BlockKind.IMAGE}),
        model_id="text-only-model",
    )
    assert "text-only-model" in str(err)
    assert "image" in str(err).lower()


def test_fail_fast_policy_passes_compatible_messages():
    policy = FailFastCapabilityPolicy()
    caps = ModelCapabilities(model_id="test", accepts=frozenset({BlockKind.TEXT}))
    messages = [Message(role=Role.USER, content=(TextBlock(text="hi"),))]
    result = policy.apply(messages, caps)
    assert result == messages


def test_fail_fast_policy_raises_on_unsupported():
    policy = FailFastCapabilityPolicy()
    caps = ModelCapabilities(model_id="text-only", accepts=frozenset({BlockKind.TEXT}))
    messages = [Message(role=Role.USER, content=(
        ImageBlock(source=ImageSource(url="http://x/a.png")),
    ))]
    with pytest.raises(CapabilityMismatchError) as exc_info:
        policy.apply(messages, caps)
    assert exc_info.value.model_id == "text-only"
    assert BlockKind.IMAGE in exc_info.value.unsupported


def test_fail_fast_policy_runtime_checkable():
    """CapabilityPolicy is a runtime Protocol."""
    assert isinstance(FailFastCapabilityPolicy(), CapabilityPolicy)


def test_capability_check_skips_truncation_block_and_preserves_nested_recursion():
    """Spec §5 (codex P1.3): _collect_block_kinds skips TruncationBlock at every
    depth but PRESERVES recursion into ToolResultBlock.content (nested
    image/audio/thinking still participate in accepts checks)."""
    from lingyia_core import (
        BlockKind, Message, Role, TextBlock, ToolResultBlock, ImageBlock,
        TruncationBlock,
    )
    from lingyia_core.blocks import ImageSource
    from lingyia_core.capability import _collect_block_kinds

    messages = [
        Message(role=Role.SYSTEM, content=(TruncationBlock(count=5),)),
        Message(role=Role.USER, content=(TextBlock(text="search"),)),
        Message(role=Role.USER, content=(
            ToolResultBlock(
                tool_use_id="t1",
                content=(
                    ImageBlock(source=ImageSource(url="https://example.com/img.png")),
                    TruncationBlock(count=5),
                ),
            ),
        )),
    ]

    kinds = _collect_block_kinds(messages)

    assert BlockKind.TEXT in kinds
    assert BlockKind.TOOL_RESULT in kinds
    assert BlockKind.IMAGE in kinds, (
        "Nested ImageBlock inside ToolResultBlock.content must participate in capability check"
    )
    # TruncationBlock skipped at every depth — does not contribute any kind
    assert "truncation" not in {k.value for k in kinds}
