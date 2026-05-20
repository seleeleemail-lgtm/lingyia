"""TokenAwareCompactor tests — v0.2 messages contract."""
import pytest

from lingyia_core import RunState
from lingyia_core.blocks import Role, TextBlock, ToolUseBlock, ToolResultBlock, TruncationBlock
from lingyia_core.message import Message
from lingyia_kit.compactors.token_aware import (
    TokenAwareCompactor, tiktoken_estimator, char_div4_estimator,
)


def _make_state_with_n_turns(n: int) -> RunState:
    """Build a state with n alternating USER-ASSISTANT turn pairs."""
    msgs = []
    for i in range(n):
        msgs.append(Message(role=Role.USER, content=(TextBlock(text=f"user msg {i} " * 20),)))
        msgs.append(Message(role=Role.ASSISTANT, content=(TextBlock(text=f"assistant msg {i} " * 20),)))
    return RunState(messages=msgs, run_id="x")


def test_no_compact_under_budget():
    c = TokenAwareCompactor(max_tokens=10_000, keep_last_turns=5, token_estimator=char_div4_estimator)
    state = _make_state_with_n_turns(2)
    assert not c.should_compact(state)


def test_compact_drops_oldest_preserves_recent():
    c = TokenAwareCompactor(max_tokens=200, keep_last_turns=2, token_estimator=char_div4_estimator)
    state = _make_state_with_n_turns(10)  # 20 messages
    assert c.should_compact(state)

    new_state = c.compact(state)
    assert len(new_state.messages) < len(state.messages)
    assert new_state.messages[-1] == state.messages[-1]
    assert new_state.messages[-2] == state.messages[-2]


def test_compact_preserves_system_message():
    c = TokenAwareCompactor(max_tokens=100, keep_last_turns=1, token_estimator=char_div4_estimator)
    state = _make_state_with_n_turns(10)
    state.messages.insert(0, Message(
        role=Role.SYSTEM,
        content=(TextBlock(text="you are a helpful assistant"),),
    ))
    new_state = c.compact(state)
    assert new_state.messages[0].role == Role.SYSTEM


def test_compact_keeps_tool_use_and_result_adjacency():
    """ToolUseBlock and its ToolResultBlock must stay together (or be dropped together)."""
    msgs = [
        Message(role=Role.USER, content=(TextBlock(text="hi"),)),
        Message(role=Role.ASSISTANT, content=(
            TextBlock(text="checking"),
            ToolUseBlock(id="t1", name="x", input={}),
        )),
        Message(role=Role.USER, content=(ToolResultBlock(tool_use_id="t1", content="r"),)),
        Message(role=Role.ASSISTANT, content=(TextBlock(text="done"),)),
    ]
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(max_tokens=10, keep_last_turns=1, token_estimator=char_div4_estimator)
    new_state = c.compact(state)

    has_tool_use = any(
        isinstance(b, ToolUseBlock)
        for m in new_state.messages for b in m.content
    )
    has_tool_result = any(
        isinstance(b, ToolResultBlock)
        for m in new_state.messages for b in m.content
    )
    # Either both present or both absent — never orphan tool_result
    assert has_tool_use == has_tool_result


def test_compact_inserts_truncation_marker():
    """After compaction, a synthetic SYSTEM message marks the truncation.

    Spec v0.2.0-β1a §7.1: marker is Message(SYSTEM, (TruncationBlock(count=N),)).
    """
    c = TokenAwareCompactor(max_tokens=100, keep_last_turns=1, token_estimator=char_div4_estimator)
    state = _make_state_with_n_turns(10)
    new_state = c.compact(state)

    marker_msgs = [
        m for m in new_state.messages
        if m.role == Role.SYSTEM
        and any(isinstance(b, TruncationBlock) for b in m.content)
    ]
    assert len(marker_msgs) >= 1
    # The marker's TruncationBlock must carry a positive dropped-count.
    assert marker_msgs[0].content[0].count > 0


def test_compact_returns_new_state_not_mutates_input():
    c = TokenAwareCompactor(max_tokens=100, keep_last_turns=1, token_estimator=char_div4_estimator)
    state = _make_state_with_n_turns(10)
    original_len = len(state.messages)
    new_state = c.compact(state)
    assert len(state.messages) == original_len  # input unchanged
    assert new_state is not state
