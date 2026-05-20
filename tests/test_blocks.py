"""Tests for ContentBlock discriminated union (v0.2-α §4.2)."""
import pytest

from lingyia_core.blocks import (
    Role, BlockKind, TextBlock, ToolUseBlock, ToolResultBlock,
    ImageBlock, ImageSource, AudioBlock, AudioSource, ThinkingBlock,
    block_to_dict, block_from_dict, UnknownBlockTypeError,
)


def test_role_enum_values():
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"
    with pytest.raises(ValueError):
        Role("tool")  # no TOOL role


def test_block_kind_covers_all_blocks():
    assert {bk.value for bk in BlockKind} == {
        "text", "tool_use", "tool_result", "image", "audio", "thinking",
    }


def test_text_block_round_trip():
    block = TextBlock(text="hello")
    d = block_to_dict(block)
    assert d == {"type": "text", "text": "hello"}
    restored = block_from_dict(d)
    assert restored == block


def test_tool_use_block_round_trip():
    block = ToolUseBlock(id="abc", name="weather", input={"city": "shanghai"})
    d = block_to_dict(block)
    assert d == {"type": "tool_use", "id": "abc", "name": "weather", "input": {"city": "shanghai"}}
    restored = block_from_dict(d)
    assert restored == block


def test_tool_result_block_with_str_content():
    block = ToolResultBlock(tool_use_id="abc", content="sunny 22C")
    d = block_to_dict(block)
    assert d == {"type": "tool_result", "tool_use_id": "abc", "content": "sunny 22C", "is_error": False}
    restored = block_from_dict(d)
    assert restored == block


def test_tool_result_block_with_nested_blocks():
    nested = (TextBlock(text="sunny"), TextBlock(text="22C"))
    block = ToolResultBlock(tool_use_id="abc", content=nested, is_error=False)
    d = block_to_dict(block)
    assert d["content"] == [
        {"type": "text", "text": "sunny"},
        {"type": "text", "text": "22C"},
    ]
    restored = block_from_dict(d)
    assert restored == block


def test_image_block_url_source():
    block = ImageBlock(source=ImageSource(url="https://example.com/x.png"))
    d = block_to_dict(block)
    assert d == {"type": "image", "source": {"url": "https://example.com/x.png", "data": "", "media_type": ""}}
    assert block_from_dict(d) == block


def test_image_block_data_source():
    block = ImageBlock(source=ImageSource(data="iVBORw0KG...", media_type="image/png"))
    assert block_from_dict(block_to_dict(block)) == block


def test_image_source_requires_url_or_data():
    with pytest.raises(ValueError, match="ImageSource requires url or data"):
        ImageSource()


def test_audio_source_requires_url_or_data():
    with pytest.raises(ValueError, match="AudioSource requires url or data"):
        AudioSource()


def test_thinking_block_round_trip():
    block = ThinkingBlock(thinking="let me think...", signature="abc123")
    d = block_to_dict(block)
    assert d == {"type": "thinking", "thinking": "let me think...", "signature": "abc123"}
    assert block_from_dict(d) == block


def test_unknown_block_type_raises():
    with pytest.raises(UnknownBlockTypeError) as exc_info:
        block_from_dict({"type": "video", "url": "..."})
    assert "video" in str(exc_info.value)


def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="missing required field"):
        block_from_dict({"type": "text"})  # missing "text"


def test_recursive_tool_result_depth_limit():
    # Build content nested 9 deep — should reject at depth > 8
    inner = TextBlock(text="bottom")
    block = inner
    for _ in range(9):
        block = ToolResultBlock(tool_use_id="x", content=(block,))
    with pytest.raises(ValueError, match="content depth exceeds 8"):
        block_to_dict(block)


def test_truncation_block_construction_rejects_zero_and_negative():
    """Spec §4.1 (codex P2.6): __post_init__ enforces count > 0."""
    from lingyia_core import TruncationBlock

    with pytest.raises(ValueError, match="positive integer"):
        TruncationBlock(count=0)

    with pytest.raises(ValueError, match="positive integer"):
        TruncationBlock(count=-1)

    b = TruncationBlock(count=1)
    assert b.count == 1
    assert b.type == "truncation"


def test_truncation_block_to_dict_from_dict_round_trip():
    """Spec §4.4: block_to_dict / block_from_dict preserve TruncationBlock."""
    from lingyia_core import TruncationBlock
    from lingyia_core.blocks import block_to_dict, block_from_dict

    original = TruncationBlock(count=42)
    serialized = block_to_dict(original)
    assert serialized == {"type": "truncation", "count": 42}

    restored = block_from_dict(serialized)
    assert isinstance(restored, TruncationBlock)
    assert restored.count == 42
    assert restored == original


def test_truncation_block_in_content_block_union():
    """Spec §4.2: TruncationBlock is a member of ContentBlock union."""
    import typing

    from lingyia_core import TruncationBlock
    from lingyia_core.blocks import ContentBlock

    args = typing.get_args(ContentBlock)
    assert TruncationBlock in args, (
        f"TruncationBlock must be in ContentBlock union; got {args}"
    )


def test_unknown_block_type_in_dict_raises_unknown_block_type_error():
    """Spec §4.4 (T3): block_from_dict remains closed. Unknown 'type' raises."""
    from lingyia_core.blocks import block_from_dict, UnknownBlockTypeError

    with pytest.raises(UnknownBlockTypeError, match="citation"):
        block_from_dict({"type": "citation", "text": "RFC 8259"})


def test_truncation_block_round_trip_to_dict_from_dict_via_runstate():
    """Spec §12 test #17 (codex P2.12 split): pure serialization round-trip
    preserves TruncationBlock in RunState."""
    from lingyia_core import (
        RunState, Role, Message, TextBlock, TruncationBlock,
    )

    state = RunState(
        messages=[
            Message(role=Role.SYSTEM, content=(TruncationBlock(count=42),)),
            Message(role=Role.USER, content=(TextBlock(text="hello"),)),
        ],
        run_id="t-roundtrip",
        iteration=3,
    )

    d = state.to_dict()
    restored = RunState.from_dict(d)

    assert restored.run_id == "t-roundtrip"
    assert restored.iteration == 3
    assert len(restored.messages) == 2

    marker = restored.messages[0].content[0]
    assert isinstance(marker, TruncationBlock)
    assert marker.count == 42
