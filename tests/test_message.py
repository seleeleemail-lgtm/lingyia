"""Tests for Message dataclass (v0.2-α §4.3)."""
import pytest

from lingyia_core.blocks import Role, TextBlock, ToolUseBlock, ToolResultBlock
from lingyia_core.message import Message


def test_message_basic():
    m = Message(role=Role.USER, content=(TextBlock(text="hi"),))
    assert m.role == Role.USER
    assert len(m.content) == 1


def test_message_round_trip():
    m = Message(role=Role.ASSISTANT, content=(
        TextBlock(text="let me check"),
        ToolUseBlock(id="abc", name="weather", input={"city": "shanghai"}),
    ))
    d = m.to_dict()
    assert d["role"] == "assistant"
    assert len(d["content"]) == 2
    restored = Message.from_dict(d)
    assert restored == m


def test_message_with_tool_result_in_user_role():
    """Anthropic-style: tool results live in user-role messages."""
    m = Message(role=Role.USER, content=(
        ToolResultBlock(tool_use_id="abc", content="sunny 22C"),
    ))
    assert Message.from_dict(m.to_dict()) == m


def test_message_empty_content_rejected():
    with pytest.raises(ValueError, match="Message.content must be non-empty"):
        Message(role=Role.USER, content=())


def test_message_from_dict_invalid_role():
    with pytest.raises(ValueError, match="invalid role"):
        Message.from_dict({"role": "tool", "content": [{"type": "text", "text": "x"}]})
