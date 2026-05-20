"""Regression tests for Anthropic adapter (v0.2 message contract).

Codex P1 finding: Anthropic emits ThinkingBlock but accepts didn't include
THINKING, so the next runtime turn would reject the saved thinking block.
This test pins the contract that THINKING must be in BOTH accepts and emits
for the Anthropic adapter, and that a thinking + tool_use + tool_result
continuation passes the FailFastCapabilityPolicy.
"""
from __future__ import annotations

import unittest

from lingyia_core import RunState
from lingyia_core.blocks import (
    BlockKind,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    TruncationBlock,
)
from lingyia_core.capability import FailFastCapabilityPolicy
from lingyia_core.message import Message
from lingyia_kit.adapters.anthropic import AnthropicModel


class TestAnthropicCapabilities(unittest.TestCase):
    """Anthropic capabilities must round-trip its own emitted blocks."""

    def setUp(self) -> None:
        self.model = AnthropicModel(api_key="dummy", model="claude-sonnet-4-5")

    def test_emits_subset_of_accepts(self) -> None:
        """Spec §16: an adapter must accept every block kind it can emit.
        Otherwise its own response will be rejected on the next turn.
        """
        caps = self.model.capabilities
        leakage = caps.emits - caps.accepts
        self.assertEqual(
            leakage, frozenset(),
            f"Anthropic emits blocks it cannot accept: {leakage}",
        )

    def test_accepts_thinking_block(self) -> None:
        """THINKING must be in accepts so it round-trips into the next turn."""
        self.assertIn(BlockKind.THINKING, self.model.capabilities.accepts)

    def test_thinking_then_tool_use_then_tool_result_passes_policy(self) -> None:
        """Regression: thinking + tool_use + tool_result continuation works.

        This is the lifecycle that broke before the fix: Claude returns
        ThinkingBlock + ToolUseBlock, runtime appends to history, executor
        appends user-role ToolResultBlock, capability policy runs and
        rejects the saved ThinkingBlock if THINKING is not in accepts.
        """
        messages = [
            Message(role=Role.USER, content=(TextBlock(text="solve it"),)),
            Message(role=Role.ASSISTANT, content=(
                ThinkingBlock(thinking="hmm", signature="sig"),
                ToolUseBlock(id="call_1", name="calc", input={"a": 1}),
            )),
            Message(role=Role.USER, content=(
                ToolResultBlock(tool_use_id="call_1", content="42"),
            )),
        ]
        policy = FailFastCapabilityPolicy()
        # Must not raise.
        result = policy.apply(messages, self.model.capabilities)
        self.assertEqual(result, messages)


def _resolve_system_payload(model: AnthropicModel, state: RunState):
    """Locate the Anthropic system payload via whichever accessor the adapter exposes."""
    if hasattr(model, "_build_system_payload"):
        return model._build_system_payload(state)
    if hasattr(model, "_extract_system_payload"):
        return model._extract_system_payload(state)
    if hasattr(model, "_extract_system_prompt"):
        # legacy name on the v0.2 adapter
        return model._extract_system_prompt(state.messages)
    return None


def _flatten_system_payload(payload) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return " ".join(
            (p.get("text", "") if isinstance(p, dict) else str(p)) for p in payload
        )
    return str(payload) if payload is not None else ""


def test_anthropic_adapter_flattens_truncation_to_text():
    """Spec §8.2: Message(SYSTEM, (TruncationBlock(12),)) is projected into the
    Anthropic system payload as '[earlier 12 messages omitted]'."""
    m = AnthropicModel(api_key="x", model="claude-sonnet-4-6")
    state = RunState(messages=[
        Message(role=Role.SYSTEM, content=(TruncationBlock(count=12),)),
    ])

    system_payload = _resolve_system_payload(m, state)
    assert system_payload is not None, "Could not find system payload in adapter"

    text = _flatten_system_payload(system_payload)
    assert "[earlier 12 messages omitted]" in text


def test_anthropic_adapter_mixed_system_message_concatenates_in_order():
    """Spec §8.2 (codex P2.9): mixed SYSTEM(TextBlock, TruncationBlock) preserves
    both contents in block-declaration order in the Anthropic system payload."""
    m = AnthropicModel(api_key="x", model="claude-sonnet-4-6")
    state = RunState(messages=[
        Message(role=Role.SYSTEM, content=(
            TextBlock(text="You are a researcher."),
            TruncationBlock(count=9),
        )),
    ])

    system_payload = _resolve_system_payload(m, state)
    assert system_payload is not None, "Could not find system payload in adapter"

    text = _flatten_system_payload(system_payload)
    assert "You are a researcher." in text
    assert "[earlier 9 messages omitted]" in text
    assert text.index("You are a researcher.") < text.index("[earlier 9 messages omitted]"), (
        "TextBlock must appear before TruncationBlock in system payload"
    )


if __name__ == "__main__":
    unittest.main()
