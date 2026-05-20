"""Model capability registry and policy framework.

Adapters declare what BlockKinds they accept/emit and what features they
support (streaming, parallel tools, JSON schema, etc.). Runtime queries this
before adecide() to enforce compatibility or downgrade gracefully.

Spec: v0.2-α §4.5, §4.7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

from .blocks import BlockKind, ContentBlock, ToolResultBlock, TruncationBlock
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
    """Raised by FailFastCapabilityPolicy when *input messages* contain block
    kinds not in ``model.capabilities.accepts``.

    Direction: client → model. The runtime refuses to send blocks the model
    has declared it cannot consume.
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


class CapabilityViolationError(RuntimeError):
    """Raised when a Model emits a block kind not in its declared
    ``capabilities.emits`` (spec §16).

    Direction: model → client. The adapter advertised what it can produce;
    if it returns something else, the runtime rejects the decision so the
    bug surfaces immediately instead of poisoning the transcript.
    """

    def __init__(
        self,
        emitted: frozenset[BlockKind],
        declared: frozenset[BlockKind],
        leaked: frozenset[BlockKind],
        model_id: str,
        reason: str | None = None,
    ) -> None:
        self.emitted = emitted
        self.declared = declared
        self.leaked = leaked
        self.model_id = model_id
        if reason is not None:
            # Caller supplied a domain-specific message (e.g. runtime-authored
            # block leaked from model output — spec §6.1). Keep model_id in
            # the message so failures stay attributable.
            super().__init__(f"Model {model_id!r}: {reason}")
            return
        leaked_list = sorted(b.value for b in leaked)
        declared_list = sorted(b.value for b in declared)
        super().__init__(
            f"Model {model_id!r} emitted blocks not in capabilities.emits: "
            f"{leaked_list}. Declared emits: {declared_list}."
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
    """Walk all ContentBlocks in messages, returning kinds present.

    Per spec §5 (codex P1.3): recurses into ToolResultBlock.content (which can be
    a tuple of nested ContentBlocks). TruncationBlock is skipped at every depth
    because it is runtime-authored, not a model capability.
    """
    kinds: set[BlockKind] = set()

    def visit(block: ContentBlock) -> None:
        if isinstance(block, TruncationBlock):
            return  # runtime-only, not a model capability — skip at every depth
        kinds.add(BlockKind(block.type))
        if isinstance(block, ToolResultBlock) and isinstance(block.content, tuple):
            for nested in block.content:
                visit(nested)

    for m in messages:
        for b in m.content:
            visit(b)
    return frozenset(kinds)


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
