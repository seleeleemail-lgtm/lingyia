"""Runtime: domain-agnostic orchestrator for the agent loop.

Runtime owns ``how to run``: model turn, tool execution, guard checks,
validator, interrupts, checkpoint calls, compaction, telemetry, retries.

Harness owns ``what to run``: tools, validator, guard, context builder.

v0.2-α message contract reset:
- ``arun``/``astream`` accept ``str`` (auto-wrapped as USER text Message) or
  ``list[Message]`` directly. The v0.1 ``goal=`` keyword has been removed.
- Conversation history lives on ``state.messages``. Observations/feedback
  are encoded as Messages (USER role with TextBlock or ToolResultBlock).
- ``Harness.system_prompt`` (optional) is injected as the first SYSTEM
  message before the model is called.
- A ``CapabilityPolicy`` runs before each ``model.adecide()`` so message
  block kinds are validated against ``model.capabilities`` (fail-fast by
  default).
- Tool execution converts ``ToolUseBlock`` → internal ``ToolCall`` for the
  executor, then back to ``ToolResultBlock`` for the transcript.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from .blocks import ContentBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock, TruncationBlock
from .capability import (
    CapabilityPolicy,
    CapabilityViolationError,
    FailFastCapabilityPolicy,
    ModelCapabilities,
)
from .executor import RetryPolicy, Tool, ToolExecutor
from .message import Message
from .protocols import Checkpointer, Compactor, Model, TelemetrySink
from .state import (
    Decision,
    DecisionKind,
    Interrupt,
    InterruptReason,
    RunResult,
    RunState,
    RunStatus,
    TelemetryEvent,
    ToolCall,
    ToolContext,
    ToolResult,
)


GuardFn = Callable[[Decision, RunState], "GuardResult"]
ValidatorFn = Callable[[RunState], "ValidationResult"]
ContextBuilderFn = Callable[[RunState, Sequence[Tool]], Mapping[str, Any]]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool = True
    requires_approval: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ValidationResult:
    done: bool = False
    summary: str = ""
    feedback: str = ""
    needs_human: bool = False
    question: str = ""


def _default_context_builder(state: RunState, tools: Sequence[Tool]) -> Mapping[str, Any]:
    """Default context per spec §10.

    Returns:
        - system_prompt: pulled from any SYSTEM-role message at the head of
          ``state.messages`` (joined TextBlocks). Adapters that carry the
          system prompt outside the messages array (e.g. Anthropic) can use
          this directly; adapters that read messages will see it twice
          unless they filter the SYSTEM role themselves.
        - tools_schema: JSON-schema-shaped list of tool descriptors.
        - metadata: free-form state.metadata dict (mutable copy).
    """
    system_prompt = "\n".join(
        b.text
        for m in state.messages
        if m.role == Role.SYSTEM
        for b in m.content
        if isinstance(b, TextBlock)
    )
    return {
        "system_prompt": system_prompt,
        "tools_schema": [
            {"name": t.name, "description": t.description, "input_schema": dict(t.input_schema)}
            for t in tools
        ],
        "metadata": dict(state.metadata),
    }


def _find_runtime_block(
    content: Sequence[ContentBlock],
) -> Optional[ContentBlock]:
    """Recursively scan a ContentBlock sequence for runtime-authored blocks.

    Per spec §6.0 (codex P2.7): single source of truth for runtime-authored
    block detection. Used by:
      - Model output validation (§6.1, Task 4)
      - Tool result validation (§6.2, Task 5)
      - Approval-resume validation (§6.3, Task 6)

    Currently the only runtime-authored block kind is TruncationBlock.

    Returns the first runtime-authored block found (for error reporting), or None.
    Recurses into ToolResultBlock.content when it is a tuple of ContentBlocks.
    """
    for b in content:
        if isinstance(b, TruncationBlock):
            return b
        if isinstance(b, ToolResultBlock) and isinstance(b.content, tuple):
            nested = _find_runtime_block(b.content)
            if nested is not None:
                return nested
    return None


def _default_guard(decision: Decision, state: RunState) -> GuardResult:
    return GuardResult()


def _default_validator(state: RunState) -> ValidationResult:
    """Default validator: trust the model.

    Returns ``ValidationResult(done=True)``. With no domain validator
    supplied, the runtime accepts the model's ``FINAL_ANSWER`` as the end
    of the run — the model is the authority on when its answer is final.

    v0.2-α: the validator runs ONLY at the ``FINAL_ANSWER`` gate. Tool
    results no longer trigger validation; the model decides whether to
    finalize, call more tools, or otherwise progress on each iteration.
    This mirrors LangGraph / OpenAI Agents SDK / Pydantic AI conventions
    where the validator/guardrail is a single gate over the final response.

    Domain authors override this to enforce completion criteria. An
    explicit ``ValidationResult(done=False)`` from a custom validator is a
    deliberate rejection at the FINAL_ANSWER gate that drives the loop
    forward (with optional ``feedback`` re-entered as a USER TextBlock).
    """
    return ValidationResult(done=True)


@dataclass
class Harness:
    """Domain-specific configuration: tools + validator + guard + context builder.

    Harness deliberately knows nothing about timeouts, retries, checkpointers,
    telemetry, or compaction. Those live in the Runtime.

    ``granted_permissions`` gates tool execution. The wildcard ``"*"`` allows
    every tool (default). In production, list explicit permission strings and
    decline broad grants — Tool.required_permissions is checked against this
    set before any tool runs.

    v0.2: ``system_prompt`` (if non-empty) is injected as the first SYSTEM
    message of the run when not already present. ``metadata`` is free-form
    storage adapters can read via Harness.metadata.
    """

    tools: list[Tool] = field(default_factory=list)
    validator: ValidatorFn = _default_validator
    guard: GuardFn = _default_guard
    context_builder: ContextBuilderFn = _default_context_builder
    granted_permissions: frozenset = field(default_factory=lambda: frozenset(["*"]))
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def tool_by_name(self, name: str) -> Optional[Tool]:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def permission_check(self, tool: Tool) -> Optional[str]:
        """Return None if allowed, or an error message describing the gap."""
        if "*" in self.granted_permissions:
            return None
        missing = tool.required_permissions - self.granted_permissions
        if missing:
            return f"missing permissions: {sorted(missing)}"
        return None


@dataclass
class Runtime:
    """Domain-agnostic orchestrator. Construct via ``Runtime.dev`` or ``Runtime.production``."""

    model: Model
    checkpointer: Checkpointer
    telemetry: TelemetrySink
    compactor: Compactor
    executor: ToolExecutor
    max_iterations: int = 8
    # 0 means no budget. When set, the run aborts as soon as accumulated
    # ``state.metadata["cost_usd"]`` exceeds this value.
    max_cost_usd: float = 0.0
    # Optional callback ``(ModelUsage) -> float`` that adapters can use to
    # compute cost when the adapter didn't fill ``usage.cost_usd`` itself.
    cost_estimator: Optional[Callable[[Any], float]] = None
    # v0.2: capability policy applied to messages before each adecide().
    capability_policy: CapabilityPolicy = field(default_factory=FailFastCapabilityPolicy)

    # Construction --------------------------------------------------------

    @classmethod
    def dev(
        cls,
        model: Model,
        max_iterations: int = 8,
        default_timeout_s: float = 60.0,
        default_retry: Optional[RetryPolicy] = None,
        max_cost_usd: float = 0.0,
        cost_estimator: Optional[Callable[[Any], float]] = None,
        capability_policy: Optional[CapabilityPolicy] = None,
    ) -> "Runtime":
        """Dev/test runtime with permissive defaults.

        Uses in-memory checkpointing, stdout telemetry, and no compaction.
        Not safe for production.
        """
        from .defaults.checkpointer import InMemoryCheckpointer
        from .defaults.compactor import NoOpCompactor
        from .defaults.telemetry import StdoutTelemetry

        return cls(
            model=model,
            checkpointer=InMemoryCheckpointer(),
            telemetry=StdoutTelemetry(),
            compactor=NoOpCompactor(),
            executor=ToolExecutor(
                default_timeout_s=default_timeout_s,
                default_retry=default_retry,
            ),
            max_iterations=max_iterations,
            max_cost_usd=max_cost_usd,
            cost_estimator=cost_estimator,
            capability_policy=capability_policy or FailFastCapabilityPolicy(),
        )

    @classmethod
    def production(
        cls,
        model: Model,
        checkpointer: Checkpointer,
        telemetry: TelemetrySink,
        compactor: Optional[Compactor] = None,
        default_timeout_s: float = 60.0,
        default_retry: Optional[RetryPolicy] = None,
        max_iterations: int = 8,
        max_cost_usd: float = 0.0,
        cost_estimator: Optional[Callable[[Any], float]] = None,
        capability_policy: Optional[CapabilityPolicy] = None,
    ) -> "Runtime":
        """Production runtime: fail closed on missing durability/observability."""
        if checkpointer is None:
            raise ValueError("production runtime requires an explicit checkpointer")
        if telemetry is None:
            raise ValueError("production runtime requires an explicit telemetry sink")
        from .defaults.compactor import NoOpCompactor

        return cls(
            model=model,
            checkpointer=checkpointer,
            telemetry=telemetry,
            compactor=compactor or NoOpCompactor(),
            executor=ToolExecutor(
                default_timeout_s=default_timeout_s,
                default_retry=default_retry,
            ),
            max_iterations=max_iterations,
            max_cost_usd=max_cost_usd,
            cost_estimator=cost_estimator,
            capability_policy=capability_policy or FailFastCapabilityPolicy(),
        )

    # Public API ----------------------------------------------------------

    def run(
        self,
        harness: Harness,
        messages: Union[list[Message], str, None] = None,
    ) -> RunResult:
        """Synchronous entry point. Internally drives the async loop."""
        return asyncio.run(self.arun(harness, messages))

    async def arun(
        self,
        harness: Harness,
        messages: Union[list[Message], str, None] = None,
    ) -> RunResult:
        """v0.2: accept ``str`` (auto-wrapped) or ``list[Message]``.

        The v0.1 ``goal=`` keyword has been removed in v0.2-α. Pass the seed
        positionally as a str or list[Message].
        """
        seed = self._coerce_seed(messages)
        state = RunState(messages=seed)
        return await self._continue(harness, state)

    async def astream(
        self,
        harness: Harness,
        messages: Union[list[Message], str, None] = None,
    ):
        """Run an agent and yield events as they happen.

        Yields tuples of ``(kind, payload)``:
        - ``("event", TelemetryEvent)`` for each emitted telemetry event
          (decision, tool_started, tool_completed, interrupt, ...).
        - ``("final", RunResult)`` once exactly, when the run terminates.

        UI clients can stream these to show incremental progress without
        blocking on the full run.
        """
        import asyncio as _asyncio

        queue: _asyncio.Queue = _asyncio.Queue()
        original_telemetry = self.telemetry
        try:
            self.telemetry = _ChainedSink(original_telemetry, _QueueSink(queue))
            run_task = _asyncio.create_task(self.arun(harness, messages))
            while True:
                getter = _asyncio.create_task(queue.get())
                done, _ = await _asyncio.wait(
                    [run_task, getter],
                    return_when=_asyncio.FIRST_COMPLETED,
                )
                # Drain any events that landed while we were waiting.
                if getter in done:
                    yield ("event", getter.result())
                else:
                    getter.cancel()
                if run_task.done():
                    # Drain remaining events.
                    while not queue.empty():
                        yield ("event", queue.get_nowait())
                    yield ("final", run_task.result())
                    return
        finally:
            self.telemetry = original_telemetry

    def resume(
        self,
        harness: Harness,
        state: RunState,
        approved: bool = True,
        feedback: str = "",
    ) -> RunResult:
        return asyncio.run(self.aresume(harness, state, approved, feedback))

    async def aresume(
        self,
        harness: Harness,
        state: RunState,
        approved: bool = True,
        feedback: str = "",
    ) -> RunResult:
        interrupt = state.interrupt
        state.interrupt = None

        if interrupt is None:
            # Plain resume with no pending interrupt — treat feedback (if any)
            # as a fresh user turn.
            if feedback:
                state.messages.append(_user_text_message(feedback))
            return await self._continue(harness, state)

        if interrupt.reason == InterruptReason.APPROVAL:
            if not approved:
                reason = feedback or "approval rejected"
                state.messages.append(
                    _user_text_message(f"User rejected the action: {reason}")
                )
                state.iteration += 1
                return await self._continue(harness, state)
            if interrupt.pending_decision is None:
                state.messages.append(
                    _user_text_message("approval interrupt missing pending decision")
                )
                state.iteration += 1
                return await self._continue(harness, state)
            # Approval granted: record the assistant turn (the pending tool
            # call) and run it, then continue the loop.
            state.messages.append(Message(
                role=Role.ASSISTANT,
                content=interrupt.pending_decision.content,
            ))
            await self._execute_decision(harness, state, interrupt.pending_decision)
            await self._advance_iteration_after_tools(harness, state)
            return await self._continue(harness, state)

        # QUESTION resume — feedback becomes a user turn re-entering the loop.
        if feedback:
            state.messages.append(_user_text_message(feedback))
        state.iteration += 1
        return await self._continue(harness, state)

    # Loop ----------------------------------------------------------------

    async def _continue(self, harness: Harness, state: RunState) -> RunResult:
        # v0.2: every Model MUST expose a ModelCapabilities object. This is
        # a hard requirement of the contract — letting it be optional would
        # silently bypass §16 enforcement for any adapter that forgot it.
        capabilities = self._require_capabilities()

        # v0.2: inject SYSTEM message at run start if harness has system_prompt.
        if (
            harness.system_prompt
            and not any(m.role == Role.SYSTEM for m in state.messages)
        ):
            state.messages.insert(0, Message(
                role=Role.SYSTEM,
                content=(TextBlock(text=harness.system_prompt),),
            ))

        while state.iteration < self.max_iterations:
            if self.compactor.should_compact(state):
                state = self.compactor.compact(state)
                self._emit(state, TelemetryEvent(kind="compacted", iteration=state.iteration))

            context = harness.context_builder(state, harness.tools)

            # Capability check (input direction): validate that no message
            # contains block kinds unsupported by the model. The default
            # policy raises; downgrade policies may rewrite messages.
            state.messages = list(
                self.capability_policy.apply(state.messages, capabilities)
            )

            try:
                decision = await self.model.adecide(context, state, harness.tools)
            except asyncio.CancelledError:
                # Let cancellation propagate so callers can shut the loop down.
                raise
            except Exception as exc:
                self._emit(state, TelemetryEvent(
                    kind="model_error",
                    iteration=state.iteration,
                    payload={"error": str(exc)},
                ))
                return RunResult(
                    status=RunStatus.FAILED,
                    state=state,
                    reason=f"model error: {exc}",
                )

            # Runtime-authored block rejection (spec §6.1, codex P1.1): a
            # model MUST NOT emit blocks the runtime owns (currently
            # TruncationBlock — see _find_runtime_block). This MUST run
            # before _enforce_emits, because BlockKind() does not know
            # runtime-authored kinds and would otherwise raise ValueError
            # first, masking the true origin of the violation.
            self._validate_no_runtime_blocks_from_model(decision)

            # Capability check (output direction, spec §16): the model must
            # only emit block kinds it declared in capabilities.emits. This
            # catches adapter bugs (e.g. a non-Anthropic model returning a
            # ThinkingBlock) before they poison the transcript.
            self._enforce_emits(decision, capabilities)

            usage = decision.usage
            cost_delta = 0.0
            if usage is not None:
                cost_delta = float(usage.cost_usd or 0.0)
                if cost_delta == 0.0 and self.cost_estimator is not None:
                    try:
                        cost_delta = float(self.cost_estimator(usage) or 0.0)
                    except Exception:
                        cost_delta = 0.0
                if cost_delta:
                    prior = float(state.metadata.get("cost_usd", 0.0) or 0.0)
                    state.metadata["cost_usd"] = round(prior + cost_delta, 6)
                # Always accumulate token counters so callers can report on
                # usage even when pricing is unknown.
                tokens = state.metadata.setdefault("tokens", {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                })
                tokens["prompt"] += int(usage.prompt_tokens or 0)
                tokens["completion"] += int(usage.completion_tokens or 0)
                tokens["cached"] += int(usage.cached_tokens or 0)

            decision_payload: dict[str, Any] = {
                "kind": decision.kind.value,
                "tool_calls": len(decision.tool_calls),
            }
            if usage is not None:
                decision_payload.update({
                    "model_id": usage.model_id,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "cost_usd": cost_delta,
                    "cost_total_usd": state.metadata.get("cost_usd", 0.0),
                })
            self._emit(state, TelemetryEvent(
                kind="decision",
                iteration=state.iteration,
                payload=decision_payload,
            ))

            # Budget enforcement: abort the run if accumulated cost exceeds
            # the configured ceiling. Done *after* recording the decision so
            # the run's trace shows the cost that triggered the abort.
            if self.max_cost_usd > 0:
                total = float(state.metadata.get("cost_usd", 0.0) or 0.0)
                if total > self.max_cost_usd:
                    return RunResult(
                        status=RunStatus.FAILED,
                        state=state,
                        reason=f"budget exceeded: ${total:.6f} > ${self.max_cost_usd:.6f}",
                    )

            if decision.kind == DecisionKind.FINAL_ANSWER:
                # Append assistant turn so transcript is complete.
                if decision.content:
                    state.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=decision.content,
                    ))
                verdict = harness.validator(state)
                if verdict.needs_human:
                    state.interrupt = Interrupt(
                        reason=InterruptReason.QUESTION,
                        message=verdict.question,
                        iteration=state.iteration,
                    )
                    return RunResult(
                        status=RunStatus.PAUSED,
                        state=state,
                        reason=verdict.question,
                    )
                # Validator verdict drives the FINAL_ANSWER gate.
                # - done=True       → COMPLETED with the answer
                # - feedback set    → inject as USER turn, iter++, continue
                # - no feedback     → iter++, continue
                #
                # Default validator (no harness override) returns done=True,
                # so an unconfigured ``Harness()`` trusts the model. A
                # caller-supplied ``ValidationResult(done=False)`` is a
                # deliberate rejection that loops until max_iterations.
                if not verdict.done:
                    if verdict.feedback:
                        state.messages.append(_user_text_message(verdict.feedback))
                    state.iteration += 1
                    continue
                return RunResult(
                    status=RunStatus.COMPLETED,
                    state=state,
                    summary=verdict.summary or decision.text,
                )

            if decision.kind == DecisionKind.ASK_HUMAN:
                state.interrupt = Interrupt(
                    reason=InterruptReason.QUESTION,
                    message=decision.text,
                    iteration=state.iteration,
                )
                return RunResult(
                    status=RunStatus.PAUSED,
                    state=state,
                    reason=decision.text,
                )

            if decision.kind == DecisionKind.ABORT:
                return RunResult(
                    status=RunStatus.FAILED,
                    state=state,
                    reason=decision.text,
                )

            if decision.kind != DecisionKind.CALL_TOOL:
                return RunResult(
                    status=RunStatus.FAILED,
                    state=state,
                    reason=f"unknown decision kind: {decision.kind}",
                )

            guard = harness.guard(decision, state)
            if guard.requires_approval:
                state.interrupt = Interrupt(
                    reason=InterruptReason.APPROVAL,
                    message=guard.reason,
                    pending_decision=decision,
                    iteration=state.iteration,
                )
                return RunResult(
                    status=RunStatus.APPROVAL_REQUIRED,
                    state=state,
                    reason=guard.reason,
                )
            if not guard.allowed:
                state.messages.append(_user_text_message(
                    f"Guard rejected tool call: {guard.reason}"
                ))
                self._emit(state, TelemetryEvent(
                    kind="guard_denied",
                    iteration=state.iteration,
                    payload={"reason": guard.reason},
                ))
                state.iteration += 1
                continue

            # CALL_TOOL: record the assistant turn (the tool_use blocks) before
            # executing so the transcript is correctly ordered.
            state.messages.append(Message(
                role=Role.ASSISTANT,
                content=decision.content,
            ))
            await self._execute_decision(harness, state, decision)
            await self._advance_iteration_after_tools(harness, state)

        return RunResult(
            status=RunStatus.STOPPED,
            state=state,
            reason="max_iterations reached",
        )

    async def _execute_decision(
        self,
        harness: Harness,
        state: RunState,
        decision: Decision,
    ) -> None:
        """Execute all ToolUseBlocks in a Decision in parallel and append
        a single user-role Message containing one ToolResultBlock per call."""
        tool_use_blocks = list(decision.tool_calls)  # tuple[ToolUseBlock, ...]
        if not tool_use_blocks:
            return
        ctx = ToolContext(
            run_id=state.run_id,
            iteration=state.iteration,
            messages=tuple(state.messages),
            metadata=dict(state.metadata),
        )
        # Convert each ToolUseBlock → internal ToolCall for the executor.
        ordered_calls: list[ToolCall] = [_block_to_call(b) for b in tool_use_blocks]
        tasks: list[Any] = []
        for call in ordered_calls:
            tool = harness.tool_by_name(call.name)
            if tool is None:
                tasks.append(_unknown_tool_result(call))
                continue
            perm_error = harness.permission_check(tool)
            if perm_error is not None:
                self._emit(state, TelemetryEvent(
                    kind="permission_denied",
                    iteration=state.iteration,
                    payload={"tool": tool.name, "reason": perm_error},
                ))
                tasks.append(_permission_denied_result(call, perm_error))
                continue
            tasks.append(
                self.executor.execute(
                    tool,
                    call,
                    ctx,
                    on_event=lambda e: self._emit(state, e),
                )
            )
        results = await asyncio.gather(*tasks)

        # Emit one ToolResultBlock per call in a single user-role Message,
        # mirroring Anthropic's tool_result convention. Output serialization
        # follows spec §8 — see _serialize_tool_output().
        result_blocks: list[ToolResultBlock] = []
        for call, result in zip(ordered_calls, results):
            if result.ok:
                content = self._serialize_tool_output(result.output)
                # Spec §6.2 (codex P1.2 + P2.7): validate per-call AFTER
                # serialization so the tuple has survived (TruncationBlock is
                # in the serializer allowlist per Task 3). Validation runs
                # inside the zip loop so each parallel tool call gets its
                # own attributed error.
                self._validate_tool_result(content, call.name)
            else:
                content = result.error or "tool error"
            result_blocks.append(ToolResultBlock(
                tool_use_id=call.call_id,
                content=content,
                is_error=not result.ok,
            ))
        state.messages.append(Message(
            role=Role.USER,
            content=tuple(result_blocks),
        ))

    @staticmethod
    def _serialize_tool_output(payload: Any) -> Any:
        """Serialize a tool's raw output into ToolResultBlock.content per spec §8.

        Rules:
        - ``None`` → empty string
        - ``str`` → kept as-is
        - ``tuple[ContentBlock, ...]`` → preserved (rich content; tool author
          opted into structured output)
        - anything else (dict, list, int, dataclass, ...) → ``json.dumps(...,
          ensure_ascii=False, default=str)``

        Falling back to ``str()`` would turn dicts into Python repr — not
        parseable as JSON by downstream LLMs. The ``default=str`` argument
        gives a last-resort serialization for non-JSONable values (e.g.
        Decimal, datetime) instead of raising mid-loop.
        """
        import json
        from .blocks import (
            AudioBlock,
            ImageBlock,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            TruncationBlock,
        )

        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        # tuple[ContentBlock, ...] passthrough — preserves rich tool output
        # so adapters can render multiple text/image parts in a single
        # tool_result. Anything in the v0.2 ContentBlock union qualifies.
        # TruncationBlock is included so runtime-authored emissions from
        # buggy tools survive serialization and are caught by §6.2 validation
        # (Task 5) rather than getting silently stringified to JSON.
        if isinstance(payload, tuple) and payload and all(
            isinstance(b, (TextBlock, ToolUseBlock, ToolResultBlock,
                           ImageBlock, AudioBlock, ThinkingBlock,
                           TruncationBlock))
            for b in payload
        ):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Pathological case (e.g. circular ref): fall back to repr
            # rather than crash the whole run.
            return str(payload)

    async def _advance_iteration_after_tools(
        self,
        harness: Harness,
        state: RunState,
    ) -> None:
        """Advance to the next iteration after tool execution completes.

        v0.2-α design: ``harness.validator`` is invoked ONLY at the
        ``FINAL_ANSWER`` gate. Tool results trigger another model iteration
        (where the model decides whether to emit ``FINAL_ANSWER``, call more
        tools, or otherwise progress). Early termination based on tool
        outcomes is the model's responsibility — it must emit
        ``Decision.FINAL_ANSWER`` itself when its job is done. This matches
        LangGraph / OpenAI Agents SDK / Pydantic AI conventions where the
        validator/guardrail is a single gate over the final response, not a
        per-tool checkpoint.

        ``harness`` is currently unused but kept for API symmetry and to
        leave room for future per-iteration hooks without re-threading the
        call sites.
        """
        del harness  # reserved for future use; see docstring
        state.iteration += 1

    def _require_capabilities(self) -> ModelCapabilities:
        """Return ``self.model.capabilities`` or raise.

        Spec §4.5/§4.7: every Model adapter MUST advertise a real
        ``ModelCapabilities`` instance. Letting this silently default would
        bypass the §16 contract check; bugs would surface only at production
        runtime against an actual Claude/OpenAI endpoint.
        """
        caps = getattr(self.model, "capabilities", None)
        if caps is None:
            raise TypeError(
                f"Model {type(self.model).__name__} has no 'capabilities' "
                "attribute. v0.2 Models MUST expose a ModelCapabilities "
                "instance (see lingyia_core.capability.ModelCapabilities)."
            )
        if not isinstance(caps, ModelCapabilities):
            raise TypeError(
                f"Model {type(self.model).__name__}.capabilities must be a "
                f"ModelCapabilities instance, got {type(caps).__name__}."
            )
        return caps

    def _validate_no_runtime_blocks_from_model(self, decision: Decision) -> None:
        """Per spec §6.1 (codex P1.1): reject runtime-authored blocks in model output.

        MUST run BEFORE ``_enforce_emits``, because ``_enforce_emits`` calls
        ``BlockKind(block.type)`` on every block, and runtime-authored kinds
        like ``'truncation'`` are intentionally NOT in ``BlockKind`` (spec §5,
        decision 5: ``TruncationBlock`` is compactor-authored, not a model
        capability). Without this pre-check, ``_enforce_emits`` would convert
        the resulting ``ValueError`` into a generic "unknown block" error that
        does not attribute the violation to runtime-authored origin.

        Uses the shared :func:`_find_runtime_block` helper so detection logic
        stays in one place across model output (§6.1), tool results (§6.2),
        and approval-resume (§6.3).
        """
        offending = _find_runtime_block(decision.content)
        if offending is None:
            return
        capabilities = self.model.capabilities
        raise CapabilityViolationError(
            emitted=frozenset(),
            declared=capabilities.emits,
            leaked=frozenset(),
            model_id=capabilities.model_id,
            reason=(
                f"emitted a {type(offending).__name__}; runtime-authored "
                "block kinds cannot originate from model output (spec §6.1)."
            ),
        )

    def _validate_tool_result(
        self,
        serialized: Union[str, tuple[ContentBlock, ...]],
        tool_name: str,
    ) -> None:
        """Per spec §6.2 (codex P1.2 + P2.7): reject TruncationBlock anywhere in
        serialized tool output. Recurses through nested ``ToolResultBlock.content``
        via the shared :func:`_find_runtime_block` helper.

        Operates on the SERIALIZED output (after :meth:`_serialize_tool_output`),
        so the tuple has survived serialization. Per spec §13.1 step 4 the
        serializer allowlist includes ``TruncationBlock`` (Task 3) precisely so
        buggy tools don't bypass this check via stringification.
        """
        if not isinstance(serialized, tuple):
            # Plain string or other shapes cannot carry runtime ContentBlocks.
            return
        offending = _find_runtime_block(serialized)
        if offending is None:
            return
        capabilities = self.model.capabilities
        raise CapabilityViolationError(
            emitted=frozenset(),
            declared=capabilities.emits,
            leaked=frozenset(),
            model_id=capabilities.model_id,
            reason=(
                f"tool {tool_name!r} returned a {type(offending).__name__}; "
                "runtime-authored block kinds cannot originate from tool "
                "output (spec §6.2)."
            ),
        )

    @staticmethod
    def _enforce_emits(decision: Decision, capabilities: ModelCapabilities) -> None:
        """Reject Decision.content blocks outside capabilities.emits (spec §16)."""
        from .blocks import BlockKind  # local import to avoid cycle hazards

        emitted: set[BlockKind] = set()
        for block in decision.content:
            try:
                emitted.add(BlockKind(block.type))  # type: ignore[attr-defined]
            except (ValueError, AttributeError):
                # Unknown block type — treat as a violation; the adapter
                # returned something not in the v0.2 union at all.
                raise CapabilityViolationError(
                    emitted=frozenset(),
                    declared=capabilities.emits,
                    leaked=frozenset(),
                    model_id=capabilities.model_id,
                )
        leaked = frozenset(emitted) - capabilities.emits
        if leaked:
            raise CapabilityViolationError(
                emitted=frozenset(emitted),
                declared=capabilities.emits,
                leaked=leaked,
                model_id=capabilities.model_id,
            )

    def _emit(self, state: RunState, event: TelemetryEvent) -> None:
        # Tag each event with the run id so downstream sinks can correlate.
        enriched = (
            event
            if event.run_id == state.run_id
            else TelemetryEvent(
                kind=event.kind,
                iteration=event.iteration,
                timestamp=event.timestamp,
                payload=dict(event.payload),
                run_id=state.run_id,
            )
        )
        state.trace.append(enriched)
        self.telemetry.emit(enriched)

    # Helpers -------------------------------------------------------------

    @staticmethod
    def _coerce_seed(
        messages: Union[list[Message], str, None],
    ) -> list[Message]:
        """Normalize arun/run input into a list[Message].

        Accepts:
        - explicit ``messages: list[Message]``
        - ``messages: str`` (wrapped as USER TextBlock)
        - ``None`` (caller must seed state separately, e.g. via resume)
        """
        if isinstance(messages, str):
            return [_user_text_message(messages)]
        if isinstance(messages, list):
            return list(messages)
        if messages is None:
            return []
        raise TypeError(
            "messages must be str, list[Message], or None; got "
            f"{type(messages).__name__}"
        )


def _user_text_message(text: str) -> Message:
    return Message(role=Role.USER, content=(TextBlock(text=text),))


def _block_to_call(block: ToolUseBlock) -> ToolCall:
    return ToolCall(name=block.name, args=dict(block.input), call_id=block.id)


class _QueueSink:
    """Telemetry sink that pushes every event into an asyncio.Queue."""

    def __init__(self, queue) -> None:
        self._queue = queue

    def emit(self, event: TelemetryEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except Exception:
            # Backpressure protection: prefer dropping rather than crashing
            # the run on telemetry-only failures.
            pass


class _ChainedSink:
    """Fan-out sink that forwards every event to both inner sinks."""

    def __init__(self, primary: TelemetrySink, secondary: TelemetrySink) -> None:
        self._primary = primary
        self._secondary = secondary

    def emit(self, event: TelemetryEvent) -> None:
        try:
            self._primary.emit(event)
        except Exception:
            pass
        try:
            self._secondary.emit(event)
        except Exception:
            pass


async def _unknown_tool_result(call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_name=call.name,
        call_id=call.call_id,
        ok=False,
        error=f"unknown tool: {call.name}",
    )


async def _permission_denied_result(call: ToolCall, reason: str) -> ToolResult:
    return ToolResult(
        tool_name=call.name,
        call_id=call.call_id,
        ok=False,
        error=f"permission denied: {reason}",
    )
