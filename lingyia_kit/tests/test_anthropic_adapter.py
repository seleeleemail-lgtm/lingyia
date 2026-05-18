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


if __name__ == "__main__":
    unittest.main()
