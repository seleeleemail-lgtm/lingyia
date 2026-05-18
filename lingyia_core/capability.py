"""Model capability registry and policy framework.

Adapters declare what BlockKinds they accept/emit and what features they
support (streaming, parallel tools, JSON schema, etc.). Runtime queries this
before adecide() to enforce compatibility or downgrade gracefully.

Spec: v0.2-α §4.5, §4.7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

from .blocks import BlockKind, ContentBlock
from .message import Message


@dataclass(frozen=True)
class ModelCapabilities:
    """What a Model adapter advertises it can do.

    Adapters expose this as a Model.capabilities property. Runtime uses
    it to enforce CapabilityPolicy before each adecide() call.
    """
    model_id: str
    accepts: frozenset[BlockKind] = field(default_factory=lambda: frozenset({BlockKind.TEXT}))
    emits: frozenset[BlockKind] = field(default_factory=lambda: frozenset({BlockKind.TEXT}))
    supports_streaming: bool = True
    supports_parallel_tools: bool = True
    supports_json_schema: bool = False
    supports_strict_schema: bool = False
    supports_prompt_caching: bool = False
    max_context_tokens: int = 8192
    max_output_tokens: int = 4096


class CapabilityMismatchError(RuntimeError):
    """Raised by FailFastCapabilityPolicy when messages contain block kinds
    not in model.capabilities.accepts.
    """

    def __init__(
        self,
        required: frozenset[BlockKind],
        accepted: frozenset[BlockKind],
        unsupported: frozenset[BlockKind],
        model_id: str,
    ) -> None:
        self.required = required
        self.accepted = accepted
        self.unsupported = unsupported
        self.model_id = model_id
        unsupported_list = sorted(b.value for b in unsupported)
        accepted_list = sorted(b.value for b in accepted)
        super().__init__(
            f"Model {model_id!r} does not accept blocks: {unsupported_list}. "
            f"Accepted: {accepted_list}."
        )


@runtime_checkable
class CapabilityPolicy(Protocol):
    """How to handle capability mismatch before model.adecide().

    Implementations either raise (fail fast) or return a modified message
    list with unsupported blocks stripped/transformed. Downgrade policies
    MUST emit a TelemetryEvent(kind='capability_downgraded', ...).
    """

    def apply(
        self,
        messages: list[Message],
        capabilities: ModelCapabilities,
    ) -> list[Message]:
        ...


def _collect_block_kinds(messages: Iterable[Message]) -> frozenset[BlockKind]:
    """Collect all BlockKinds present in messages. Recurses into ToolResultBlock content."""
    found: set[BlockKind] = set()

    def visit(block: ContentBlock) -> None:
        type_name = block.type  # type: ignore[attr-defined]
        try:
            found.add(BlockKind(type_name))
        except ValueError:
            pass
        # Recurse into nested ToolResultBlock content
        if type_name == "tool_result":
            content = block.content  # type: ignore[attr-defined]
            if not isinstance(content, str):
                for nested in content:
                    visit(nested)

    for msg in messages:
        for block in msg.content:
            visit(block)

    return frozenset(found)


class FailFastCapabilityPolicy:
    """Default policy: raise CapabilityMismatchError if any block kind unsupported."""

    def apply(
        self,
        messages: list[Message],
        capabilities: ModelCapabilities,
    ) -> list[Message]:
        required = _collect_block_kinds(messages)
        unsupported = required - capabilities.accepts
        if unsupported:
            raise CapabilityMismatchError(
                required=required,
                accepted=capabilities.accepts,
                unsupported=unsupported,
                model_id=capabilities.model_id,
            )
        return messages
