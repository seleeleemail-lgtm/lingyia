"""Typed runtime state for the agent loop.

All types here are JSON-serializable to support cross-process pause/resume.
This module owns the data shapes; the loop and runtime own the transitions.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional
from uuid import uuid4


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
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    tool_calls: tuple[ToolCall, ...] = ()
    content: str = ""

    @classmethod
    def call_tool(cls, name: str, args: Optional[Mapping[str, Any]] = None) -> "Decision":
        return cls(
            kind=DecisionKind.CALL_TOOL,
            tool_calls=(ToolCall(name=name, args=dict(args or {})),),
        )

    @classmethod
    def call_tools(cls, tool_calls: tuple[ToolCall, ...]) -> "Decision":
        if not tool_calls:
            raise ValueError("call_tools requires at least one ToolCall")
        return cls(kind=DecisionKind.CALL_TOOL, tool_calls=tool_calls)

    @classmethod
    def final_answer(cls, content: str) -> "Decision":
        return cls(kind=DecisionKind.FINAL_ANSWER, content=content)

    @classmethod
    def ask_human(cls, question: str) -> "Decision":
        return cls(kind=DecisionKind.ASK_HUMAN, content=question)

    @classmethod
    def abort(cls, reason: str) -> "Decision":
        return cls(kind=DecisionKind.ABORT, content=reason)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    call_id: str = ""
    ok: bool = True
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    attempts: int = 1


@dataclass(frozen=True)
class Observation:
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


@dataclass(frozen=True)
class Interrupt:
    reason: InterruptReason
    message: str
    pending_decision: Optional[Decision] = None
    iteration: int = 0


@dataclass(frozen=True)
class ToolContext:
    """Read-only view passed to tool handlers.

    Tools must not mutate run state directly. They return ToolResult; the runtime
    integrates results into state. Note: this does not prevent side effects on
    external systems. Replayability of side-effecting tools requires idempotency
    keys and a durable action log, not captured here.
    """

    run_id: str
    iteration: int
    goal: str
    metadata: Mapping[str, Any]


@dataclass
class RunState:
    """Mutable per-run state. Always JSON-serializable via to_dict()."""

    goal: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    iteration: int = 0
    observations: list[Observation] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    trace: list[TelemetryEvent] = field(default_factory=list)
    interrupt: Optional[Interrupt] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)

    def add_feedback(self, feedback: str) -> None:
        if feedback:
            self.feedback.append(feedback)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "observations": [asdict(o) for o in self.observations],
            "feedback": list(self.feedback),
            "trace": [asdict(t) for t in self.trace],
            "interrupt": asdict(self.interrupt) if self.interrupt else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    state: RunState
    summary: str = ""
    reason: str = ""
