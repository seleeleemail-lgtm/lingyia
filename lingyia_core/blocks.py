"""ContentBlock discriminated union for v0.2 message contract.

Dataclass + `type` literal discriminator. No Pydantic — preserves lingyia's
"zero hard deps beyond httpx" philosophy. Serialization via `block_to_dict`
+ `block_from_dict` factory dispatching on `type` field.

Spec: v0.2-α §4.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Union


class Role(str, Enum):
    """Conversation roles. No TOOL role (Anthropic style)."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class BlockKind(str, Enum):
    """Capability declaration enum — what BlockKinds a Model accepts/emits."""
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    AUDIO = "audio"
    THINKING = "thinking"


_MAX_CONTENT_DEPTH = 8


class UnknownBlockTypeError(ValueError):
    """Raised when block_from_dict encounters an unknown `type` field."""


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: Mapping[str, Any] = field(default_factory=dict)
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class ImageSource:
    """One of: url (remote), or data+media_type (inline base64)."""
    url: str = ""
    data: str = ""
    media_type: str = ""

    def __post_init__(self) -> None:
        if not self.url and not (self.data and self.media_type):
            raise ValueError("ImageSource requires url or data+media_type")


@dataclass(frozen=True)
class ImageBlock:
    source: ImageSource
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class AudioSource:
    url: str = ""
    data: str = ""
    media_type: str = ""

    def __post_init__(self) -> None:
        if not self.url and not (self.data and self.media_type):
            raise ValueError("AudioSource requires url or data+media_type")


@dataclass(frozen=True)
class AudioBlock:
    source: AudioSource
    type: Literal["audio"] = "audio"


@dataclass(frozen=True)
class ThinkingBlock:
    """Reasoning trace (Anthropic-specific). Not user-visible text.
    Persisted in checkpoint so resume works for signed thinking.
    """
    thinking: str
    signature: str = ""
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True)
class ToolResultBlock:
    """Result of a tool execution.

    `content` is either a str (provider-agnostic flat result) or a tuple of
    ContentBlocks (for rich nested results, e.g. multi-modal tool outputs).
    Depth limited to 8 to prevent malformed cycles.
    """
    tool_use_id: str
    content: Union[str, "tuple[ContentBlock, ...]"] = ""
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = Union[
    TextBlock, ToolUseBlock, ToolResultBlock,
    ImageBlock, AudioBlock, ThinkingBlock,
]


def block_to_dict(block: ContentBlock, _depth: int = 0) -> dict[str, Any]:
    """Serialize a ContentBlock to a JSON-ready dict. Recurses into nested ToolResultBlock content."""
    if _depth > _MAX_CONTENT_DEPTH:
        raise ValueError(f"content depth exceeds {_MAX_CONTENT_DEPTH}")

    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
    if isinstance(block, ToolResultBlock):
        if isinstance(block.content, str):
            content_payload = block.content
        else:
            content_payload = [block_to_dict(b, _depth + 1) for b in block.content]
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": content_payload,
            "is_error": block.is_error,
        }
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {
                "url": block.source.url,
                "data": block.source.data,
                "media_type": block.source.media_type,
            },
        }
    if isinstance(block, AudioBlock):
        return {
            "type": "audio",
            "source": {
                "url": block.source.url,
                "data": block.source.data,
                "media_type": block.source.media_type,
            },
        }
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}

    raise TypeError(f"unknown ContentBlock subtype: {type(block).__name__}")


def block_from_dict(data: Mapping[str, Any], _depth: int = 0) -> ContentBlock:
    """Deserialize a ContentBlock dict back to its dataclass. Dispatches on `type` field."""
    if _depth > _MAX_CONTENT_DEPTH:
        raise ValueError(f"content depth exceeds {_MAX_CONTENT_DEPTH}")

    block_type = data.get("type")
    if block_type is None:
        raise ValueError("missing required field: type")

    if block_type == "text":
        if "text" not in data:
            raise ValueError("missing required field: text")
        return TextBlock(text=data["text"])

    if block_type == "tool_use":
        for required in ("id", "name"):
            if required not in data:
                raise ValueError(f"missing required field: {required}")
        return ToolUseBlock(
            id=data["id"],
            name=data["name"],
            input=dict(data.get("input", {})),
        )

    if block_type == "tool_result":
        if "tool_use_id" not in data:
            raise ValueError("missing required field: tool_use_id")
        raw_content = data.get("content", "")
        if isinstance(raw_content, str):
            content: Union[str, tuple[ContentBlock, ...]] = raw_content
        else:
            content = tuple(block_from_dict(b, _depth + 1) for b in raw_content)
        return ToolResultBlock(
            tool_use_id=data["tool_use_id"],
            content=content,
            is_error=data.get("is_error", False),
        )

    if block_type == "image":
        src = data.get("source", {})
        return ImageBlock(source=ImageSource(
            url=src.get("url", ""),
            data=src.get("data", ""),
            media_type=src.get("media_type", ""),
        ))

    if block_type == "audio":
        src = data.get("source", {})
        return AudioBlock(source=AudioSource(
            url=src.get("url", ""),
            data=src.get("data", ""),
            media_type=src.get("media_type", ""),
        ))

    if block_type == "thinking":
        if "thinking" not in data:
            raise ValueError("missing required field: thinking")
        return ThinkingBlock(
            thinking=data["thinking"],
            signature=data.get("signature", ""),
        )

    raise UnknownBlockTypeError(f"unknown block type: {block_type!r}")
