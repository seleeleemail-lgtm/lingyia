"""Message dataclass — role + content blocks. Source of truth for conversation.

Spec: v0.2-α §4.3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .blocks import ContentBlock, Role, block_from_dict, block_to_dict


@dataclass(frozen=True)
class Message:
    """One turn of the conversation. Tuple of ContentBlocks (text, tool_use, etc.)."""
    role: Role
    content: tuple[ContentBlock, ...]

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Message.content must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": [block_to_dict(b) for b in self.content],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Message":
        try:
            role = Role(data["role"])
        except (ValueError, KeyError) as exc:
            raise ValueError(f"invalid role: {data.get('role')!r}") from exc
        return cls(
            role=role,
            content=tuple(block_from_dict(b) for b in data.get("content", [])),
        )
