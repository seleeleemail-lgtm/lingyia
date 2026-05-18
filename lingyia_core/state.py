"""Typed runtime state for the v0.2 agent loop.

v0.2-α breaking changes from v0.1:
- RunState.goal removed; messages: list[Message] is source of truth
- RunState.observations removed; tool results encoded as ToolResultBlock
- RunState.feedback removed; encoded as user-role Message with TextBlock
- RunState schema_version bumped 1 → 2; v0.1 snapshots rejected
- Decision.content: str → tuple[ContentBlock, ...]
- Decision.tool_calls becomes derived accessor (filter ToolUseBlock)
- ToolContext.goal removed; messages: tuple[Message, ...] added

Spec: v0.2-α §4.8, §4.9, §6
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional
from uuid import uuid4

from .blocks import (
    ContentBlock,
    TextBlock,
    ToolUseBlock,
    block_from_dict,
    block_to_dict,
)
from .message import Message


class DecisionKind(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"
    ASK_HUMAN = "ask_human"
    ABORT = "abort"


class InterruptReason(str, Enum):
    APPROVAL = "approval"
    QUESTION = "question"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    STOPPED = "stopped"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class ToolCall:
    """Internal executor representation. NOT part of public transcript.

    Runtime converts Decision.tool_calls (ToolUseBlock) → ToolCall for executor.
    Existing tool handlers continue to consume ToolCall unchanged.
    """
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class ModelUsage:
    """Token usage and cost for one model invocation. v0.1 contract unchanged."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    model_id: str = ""


@dataclass(frozen=True)
class Decision:
    """Model output: control signal + content blocks + optional usage.

    v0.2 change: content is now tuple[ContentBlock, ...] (was str).
    tool_calls becomes derived from content. ASK_HUMAN/ABORT store their
    text in TextBlock content so decision.text accessor still works.
    """
    kind: DecisionKind
    content: tuple[ContentBlock, ...] = ()
    usage: Optional[ModelUsage] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DecisionKind):
            object.__setattr__(self, "kind", DecisionKind(self.kind))

    @property
    def tool_calls(self) -> tuple[ToolUseBlock, ...]:
        """Derived: all ToolUseBlocks in content."""
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))

    @property
    def text(self) -> str:
        """Concatenated user-visible text. Does NOT include ThinkingBlock content."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @classmethod
    def call_tools(cls, blocks) -> "Decision":
        block_tuple = tuple(blocks)
        if not block_tuple:
            raise ValueError("call_tools requires at least one ToolUseBlock")
        for b in block_tuple:
            if not isinstance(b, ToolUseBlock):
                raise TypeError(f"call_tools expects ToolUseBlock, got {type(b).__name__}")
        return cls(kind=DecisionKind.CALL_TOOL, content=block_tuple)

    @classmethod
    def final_answer(cls, content) -> "Decision":
        """Build a FINAL_ANSWER decision. Accepts str (wrapped as TextBlock) or sequence of blocks."""
        if isinstance(content, str):
            return cls(kind=DecisionKind.FINAL_ANSWER, content=(TextBlock(text=content),))
        return cls(kind=DecisionKind.FINAL_ANSWER, content=tuple(content))

    @classmethod
    def ask_human(cls, question: str) -> "Decision":
        return cls(kind=DecisionKind.ASK_HUMAN, content=(TextBlock(text=question),))

    @classmethod
    def abort(cls, reason: str) -> "Decision":
        return cls(kind=DecisionKind.ABORT, content=(TextBlock(text=reason),))


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool invocation. v0.1 contract unchanged.

    Runtime converts ToolResult → ToolResultBlock when appending to messages
    (see v0.2-α §8).
    """
    tool_name: str
    ok: bool = True
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    attempts: int = 1
    call_id: str = ""


@dataclass(frozen=True)
class Observation:
    """DEPRECATED in v0.2. Kept exported for migration tooling only.

    v0.1 used observations to record tool results and intermediate state.
    v0.2 uses messages as source of truth. Migration helper at
    lingyia_kit.migrations.v01_to_v02 converts v0.1 observations into
    appropriate v0.2 messages.
    """
    iteration: int
    kind: str
    payload: Any
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TelemetryEvent:
    kind: str
    iteration: int
    timestamp: float = field(default_factory=time.time)
    payload: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = ""


@dataclass(frozen=True)
class Interrupt:
    reason: InterruptReason
    message: str
    pending_decision: Optional[Decision] = None
    iteration: int = 0


@dataclass(frozen=True)
class ToolContext:
    """Read-only view passed to tool handlers.

    v0.2 change: no `goal` field. Tools that want "current user query"
    should extract from latest USER-role Message TextBlock in `messages`.
    """
    run_id: str
    iteration: int
    messages: tuple[Message, ...]
    metadata: Mapping[str, Any]


# v0.2-α schema version bump. v0.1 (schema_version=1) snapshots are rejected
# at load time; use lingyia_kit.migrations.v01_to_v02 to upgrade.
RUN_STATE_SCHEMA_VERSION = 2


@dataclass
class RunState:
    """Mutable per-run state. v0.2 contract.

    messages is the conversation source of truth. Tool results live as
    ToolResultBlock inside user-role messages (Anthropic style).
    Trace events still recorded separately for telemetry.
    """
    messages: list[Message]
    run_id: str = field(default_factory=lambda: str(uuid4()))
    iteration: int = 0
    trace: list[TelemetryEvent] = field(default_factory=list)
    interrupt: Optional[Interrupt] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = RUN_STATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "messages": [m.to_dict() for m in self.messages],
            "run_id": self.run_id,
            "iteration": self.iteration,
            "trace": [asdict(t) for t in self.trace],
            "interrupt": _interrupt_to_dict(self.interrupt) if self.interrupt else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunState":
        """Load v0.2 snapshot. Rejects BOTH v0.1 (version < 2) and future (version > 2).

        For v0.1 snapshots, use lingyia_kit.migrations.v01_to_v02.migrate_state_dict()
        to upgrade before loading.
        """
        version = data.get("schema_version", RUN_STATE_SCHEMA_VERSION)
        if version != RUN_STATE_SCHEMA_VERSION:
            raise UnknownSchemaVersionError(
                f"snapshot schema_version={version} not loadable by runtime "
                f"version={RUN_STATE_SCHEMA_VERSION}. Use "
                f"lingyia_kit.migrations.v01_to_v02 for v0.1 → v0.2 migration."
            )

        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        trace = [
            TelemetryEvent(
                kind=t["kind"],
                iteration=t["iteration"],
                timestamp=t.get("timestamp", time.time()),
                payload=dict(t.get("payload", {})),
                run_id=t.get("run_id", ""),
            )
            for t in data.get("trace", [])
        ]
        interrupt = _interrupt_from_dict(data.get("interrupt"))
        return cls(
            messages=messages,
            run_id=data["run_id"],
            iteration=data.get("iteration", 0),
            trace=trace,
            interrupt=interrupt,
            metadata=dict(data.get("metadata", {})),
            schema_version=version,
        )


class UnknownSchemaVersionError(ValueError):
    """Raised when a checkpoint's schema_version isn't loadable.

    v0.2-α: raised for both version < 2 and version > 2.
    """


def _interrupt_to_dict(interrupt: Interrupt) -> dict[str, Any]:
    pending: Optional[dict[str, Any]] = None
    if interrupt.pending_decision is not None:
        pd = interrupt.pending_decision
        pending = {
            "kind": pd.kind.value,
            "content": [block_to_dict(b) for b in pd.content],
        }
    return {
        "reason": interrupt.reason.value,
        "message": interrupt.message,
        "pending_decision": pending,
        "iteration": interrupt.iteration,
    }


def _interrupt_from_dict(raw: Optional[Mapping[str, Any]]) -> Optional[Interrupt]:
    if not raw:
        return None
    pending_raw = raw.get("pending_decision")
    pending: Optional[Decision] = None
    if pending_raw:
        content_blocks = tuple(block_from_dict(b) for b in pending_raw.get("content", []))
        pending = Decision(
            kind=DecisionKind(pending_raw["kind"]),
            content=content_blocks,
        )
    return Interrupt(
        reason=InterruptReason(raw["reason"]),
        message=raw["message"],
        pending_decision=pending,
        iteration=raw.get("iteration", 0),
    )


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state: RunState
    summary: str = ""
    reason: str = ""
