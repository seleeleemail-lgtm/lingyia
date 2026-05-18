"""Public protocols for the agent runtime.

Core owns the protocol contracts and trigger points. Concrete backends
(Redis, OTel, OpenAI, Anthropic) live outside this package or in defaults/.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .capability import ModelCapabilities
from .state import Decision, RunState, TelemetryEvent


@runtime_checkable
class ToolSchema(Protocol):
    """Self-describing tool metadata. Provider adapters translate to native formats."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@runtime_checkable
class Model(Protocol):
    """A reasoning engine that produces Decisions given the current run context.

    Implementations are async. Provider adapters (Claude, OpenAI, local) wrap
    their native APIs and return a Decision per invocation.
    """

    @property
    def capabilities(self) -> "ModelCapabilities": ...

    async def adecide(
        self,
        context: Mapping[str, Any],
        state: RunState,
        tools: Sequence[ToolSchema],
    ) -> Decision: ...


@runtime_checkable
class Checkpointer(Protocol):
    """Durable storage for RunState across processes.

    Implementations must accept the full RunState and return it intact.
    JSON serialization is recommended for portability.
    """

    async def asave(self, run_id: str, state: RunState) -> None: ...

    async def aload(self, run_id: str) -> Optional[RunState]: ...


@runtime_checkable
class Compactor(Protocol):
    """Context window management. Trigger and policy live with the impl."""

    def should_compact(self, state: RunState) -> bool: ...

    def compact(self, state: RunState) -> RunState: ...


@runtime_checkable
class TelemetrySink(Protocol):
    """Consumes typed telemetry events emitted by the runtime.

    Production sinks should fan out to metrics, logs, and traces. The runtime
    emits events; sinks decide where they go.
    """

    def emit(self, event: TelemetryEvent) -> None: ...
