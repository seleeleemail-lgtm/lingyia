"""Core agent loop tests — v0.2 messages contract.

Rewritten in Task 6 of the v0.2-α reset plan. These tests assert the v0.2
shape (messages as source of truth, ToolResultBlock for tool output, etc.).

Currently RED until Phase 3 runtime migration (Task 8) completes — Runtime
still constructs state via the v0.1 ``goal=`` keyword. That's expected per
the plan; do not "fix" by reverting tests.

Decision/RunState-only tests SHOULD pass because Task 4 already shipped
the v0.2 state shape. If they don't, that's a regression worth flagging.

Test names track 1:1 with the v0.1 test names for traceability.
"""
from __future__ import annotations

import json

import pytest

from lingyia_core import (
    Decision,
    DecisionKind,
    GuardResult,
    Harness,
    Interrupt,
    InterruptReason,
    RetryPolicy,
    RunResult,
    RunState,
    RunStatus,
    Runtime,
    Tool,
    ToolContext,
    ToolResult,
    ValidationResult,
)
from lingyia_core.blocks import (
    BlockKind,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from lingyia_core.capability import ModelCapabilities
from lingyia_core.message import Message
from lingyia_core.state import UnknownSchemaVersionError


# ---------------------------------------------------------------------------
# Model fakes
# ---------------------------------------------------------------------------


_DEFAULT_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
    emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
)


class ScriptedModel:
    """Yields a pre-recorded sequence of Decisions."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return _DEFAULT_CAPS

    async def adecide(self, context, state, tools):
        if not self.decisions:
            raise AssertionError("model was called more times than expected")
        self.calls += 1
        return self.decisions.pop(0)


class RepeatingModel:
    def __init__(self, decision: Decision):
        self.decision = decision

    @property
    def capabilities(self) -> ModelCapabilities:
        return _DEFAULT_CAPS

    async def adecide(self, context, state, tools):
        return self.decision


class CrashingModel:
    @property
    def capabilities(self) -> ModelCapabilities:
        return _DEFAULT_CAPS

    async def adecide(self, context, state, tools):
        raise RuntimeError("provider down")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime(model, **kw) -> Runtime:
    from lingyia_core.defaults.telemetry import NoopTelemetry

    runtime = Runtime.dev(model=model, **kw)
    runtime.telemetry = NoopTelemetry()
    return runtime


def _sync_tool(name: str, handler) -> Tool:
    return Tool.from_sync(name=name, description=name, handler=handler)


def _async_tool(name: str, handler) -> Tool:
    return Tool.from_async(name=name, description=name, handler=handler)


def _call_tool(name: str, args=None, call_id: str = "t1") -> Decision:
    """Build a single-tool CALL_TOOL Decision (v0.2 wraps as ToolUseBlock)."""
    return Decision.call_tools([
        ToolUseBlock(id=call_id, name=name, input=dict(args or {})),
    ])


def _tool_result_blocks(state: RunState) -> list[ToolResultBlock]:
    """Collect every ToolResultBlock across user-role messages."""
    out: list[ToolResultBlock] = []
    for msg in state.messages:
        if msg.role != Role.USER:
            continue
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                out.append(block)
    return out


def _feedback_texts(state: RunState) -> list[str]:
    """Collect user-role TextBlock texts that are NOT the original goal.

    Mirrors v0.1 ``state.feedback`` — every user TextBlock after the first
    user message is treated as feedback fed back into the loop.
    """
    out: list[str] = []
    first_user_seen = False
    for msg in state.messages:
        if msg.role != Role.USER:
            continue
        if not first_user_seen:
            first_user_seen = True
            continue
        for block in msg.content:
            if isinstance(block, TextBlock):
                out.append(block.text)
    return out


# ---------------------------------------------------------------------------
# 1. Tool-driven completion
# ---------------------------------------------------------------------------


async def test_tool_result_satisfies_validator():
    """Tool-call → ToolResult → ToolResultBlock appended as user-role message.

    v0.2-α: the validator is gated on FINAL_ANSWER only — after tool
    execution the loop continues and asks the model again. The model here
    emits a FINAL_ANSWER on its second turn, where the validator's
    ``done=True`` is honored.
    """
    seen: list = []

    def echo(args, ctx: ToolContext) -> ToolResult:
        seen.append(args)
        return ToolResult(tool_name="echo", ok=True, output={"seen": args["text"]})

    harness = Harness(
        tools=[_sync_tool("echo", echo)],
        validator=lambda state: ValidationResult(done=True, summary="done"),
    )
    result = await _runtime(ScriptedModel(
        _call_tool("echo", {"text": "hello"}),
        Decision.final_answer("echoed"),
    )).arun(
        harness, "echo hello"
    )

    assert result.status == RunStatus.COMPLETED
    assert result.summary == "done"
    assert seen == [{"text": "hello"}]

    # v0.2: tool result is encoded as a ToolResultBlock inside a user message.
    tool_blocks = _tool_result_blocks(result.state)
    assert len(tool_blocks) == 1
    tr = tool_blocks[-1]
    assert tr.tool_use_id == "t1"
    assert not tr.is_error
    # Content carries the tool output. Implementations may store it as a
    # serialized string or as nested blocks; both should mention "hello".
    rendered = tr.content if isinstance(tr.content, str) else json.dumps(
        [b.__dict__ for b in tr.content], default=str
    )
    assert "hello" in rendered


# ---------------------------------------------------------------------------
# 2. Direct final answer
# ---------------------------------------------------------------------------


async def test_final_answer_stops_loop():
    """FINAL_ANSWER terminates without any tool execution."""
    result = await _runtime(ScriptedModel(Decision.final_answer("ready"))).arun(
        Harness(), "answer directly"
    )
    assert result.status == RunStatus.COMPLETED
    assert result.summary == "ready"
    # No ToolResultBlocks were appended (no tools ran).
    assert _tool_result_blocks(result.state) == []


# ---------------------------------------------------------------------------
# 3. Approval pause + resume
# ---------------------------------------------------------------------------


async def test_approval_pause_and_resume():
    """Guard requires_approval pauses run; resume(approved=True) executes pending tool."""
    calls: list = []

    def risky(args, ctx):
        calls.append(args)
        return ToolResult(tool_name="risky", ok=True, output="applied")

    def validator(state: RunState) -> ValidationResult:
        done = any(
            tr.tool_use_id and not tr.is_error
            for tr in _tool_result_blocks(state)
        )
        return ValidationResult(done=done, summary="approved action completed")

    harness = Harness(
        tools=[_sync_tool("risky", risky)],
        guard=lambda decision, state: GuardResult(
            requires_approval=True, reason="review required"
        ),
        validator=validator,
    )
    # v0.2-α: validator fires only at FINAL_ANSWER. After the approval is
    # granted and the tool runs, the model emits FINAL_ANSWER so the
    # validator gate can complete the run.
    runtime = _runtime(ScriptedModel(
        _call_tool("risky", {"id": 1}),
        Decision.final_answer("applied risky action"),
    ))

    paused = await runtime.arun(harness, "do risky thing")
    assert paused.status == RunStatus.APPROVAL_REQUIRED
    assert paused.reason == "review required"
    assert paused.state.interrupt is not None
    assert paused.state.interrupt.reason == InterruptReason.APPROVAL
    assert paused.state.interrupt.pending_decision is not None

    resumed = await runtime.aresume(harness, paused.state, approved=True)
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.summary == "approved action completed"
    assert calls == [{"id": 1}]


async def test_approval_rejection_feeds_back():
    """Resume(approved=False, feedback=...) records feedback as a user message and continues."""
    runtime = _runtime(ScriptedModel(
        _call_tool("noop"),
        Decision.final_answer("gave up"),
    ))

    def noop(args, ctx):
        return ToolResult(tool_name="noop", ok=True, output=None)

    harness = Harness(
        tools=[_sync_tool("noop", noop)],
        guard=lambda decision, state: GuardResult(
            requires_approval=True, reason="blocked"
        ),
    )
    paused = await runtime.arun(harness, "should be blocked")
    assert paused.status == RunStatus.APPROVAL_REQUIRED

    resumed = await runtime.aresume(
        harness, paused.state, approved=False, feedback="too risky"
    )
    assert resumed.status == RunStatus.COMPLETED

    # v0.2: rejection feedback shows up as user-role TextBlock content.
    fb = _feedback_texts(resumed.state)
    assert any("too risky" in t for t in fb)


# ---------------------------------------------------------------------------
# 4. Tool error retried via feedback
# ---------------------------------------------------------------------------


async def test_tool_error_becomes_feedback():
    """Failing tool yields ToolResultBlock(is_error=True); model sees error and retries.

    v0.2-α: validator fires only at FINAL_ANSWER. When a tool fails, the
    error surfaces as a ToolResultBlock(is_error=True) in the transcript and
    the model — seeing the failure in its context — decides whether to
    retry, abort, or finalize. Here the scripted model retries on attempt 2
    and then emits a FINAL_ANSWER that satisfies the FINAL_ANSWER validator.
    """
    attempts: list = []

    def unstable(args, ctx):
        attempts.append(args)
        if len(attempts) == 1:
            raise RuntimeError("temporary failure")
        return ToolResult(tool_name="unstable", ok=True, output="recovered")

    def validate(state: RunState) -> ValidationResult:
        # FINAL_ANSWER gate: accept once a successful tool result is in the
        # transcript. Returning done=False here would loop until
        # max_iterations.
        blocks = _tool_result_blocks(state)
        if blocks and not blocks[-1].is_error:
            return ValidationResult(done=True, summary="recovered")
        return ValidationResult(done=False)

    harness = Harness(
        tools=[_sync_tool("unstable", unstable)],
        validator=validate,
    )
    result = await _runtime(ScriptedModel(
        _call_tool("unstable", {"attempt": 1}, call_id="t1"),
        _call_tool("unstable", {"attempt": 2}, call_id="t2"),
        Decision.final_answer("recovered"),
    )).arun(harness, "recover from tool failure")

    assert result.status == RunStatus.COMPLETED
    assert result.summary == "recovered"

    blocks = _tool_result_blocks(result.state)
    assert len(blocks) == 2
    # First attempt is the error — surfaces as ToolResultBlock(is_error=True).
    assert blocks[0].is_error is True
    err_content = (
        blocks[0].content
        if isinstance(blocks[0].content, str)
        else json.dumps([b.__dict__ for b in blocks[0].content], default=str)
    )
    assert "temporary failure" in err_content
    # Second attempt recovered.
    assert blocks[1].is_error is False


# ---------------------------------------------------------------------------
# 5. Max iterations stop
# ---------------------------------------------------------------------------


async def test_max_iterations_stop():
    """Hitting max_iterations returns STOPPED with reason set."""
    def noop(args, ctx):
        return ToolResult(tool_name="noop", ok=True, output="still running")

    harness = Harness(
        tools=[_sync_tool("noop", noop)],
        validator=lambda state: ValidationResult(feedback="not done yet"),
    )
    result = await _runtime(
        RepeatingModel(_call_tool("noop")),
        max_iterations=2,
    ).arun(harness, "never finishes")

    assert result.status == RunStatus.STOPPED
    assert result.reason == "max_iterations reached"
    assert result.state.iteration == 2
    assert len(_tool_result_blocks(result.state)) == 2


# ---------------------------------------------------------------------------
# 6. ask_human pauses then resumes with feedback
# ---------------------------------------------------------------------------


async def test_ask_human_pauses_then_resumes_with_feedback():
    """ASK_HUMAN pauses with QUESTION interrupt; resume injects feedback as user message."""
    runtime = _runtime(ScriptedModel(
        Decision.ask_human("what's the threshold?"),
        Decision.final_answer("done with answer"),
    ))
    harness = Harness()

    paused = await runtime.arun(harness, "needs human input")
    assert paused.status == RunStatus.PAUSED
    assert paused.reason == "what's the threshold?"
    assert paused.state.interrupt is not None
    assert paused.state.interrupt.reason == InterruptReason.QUESTION
    assert paused.state.interrupt.pending_decision is None

    resumed = await runtime.aresume(
        harness, paused.state, approved=True, feedback="threshold is 100"
    )
    assert resumed.status == RunStatus.COMPLETED
    assert any("threshold is 100" in t for t in _feedback_texts(resumed.state))


# ---------------------------------------------------------------------------
# 7. abort
# ---------------------------------------------------------------------------


async def test_abort_returns_failed():
    """ABORT terminates run as FAILED with reason carrying the abort message."""
    runtime = _runtime(ScriptedModel(Decision.abort("unsafe input")))
    result = await runtime.arun(Harness(), "abort me")
    assert result.status == RunStatus.FAILED
    assert result.reason == "unsafe input"


# ---------------------------------------------------------------------------
# 8. Model exception
# ---------------------------------------------------------------------------


async def test_model_exception_returns_failed_not_crash():
    """Exception inside Model.adecide is caught; run returns FAILED."""
    runtime = _runtime(CrashingModel())
    result = await runtime.arun(Harness(), "model dies")
    assert result.status == RunStatus.FAILED
    assert "provider down" in result.reason


# ---------------------------------------------------------------------------
# 9. Parallel tool calls
# ---------------------------------------------------------------------------


async def test_parallel_tool_calls_run_concurrently():
    """Multiple ToolUseBlocks in one Decision execute in parallel, one ToolResultBlock per call."""
    import asyncio

    started: list = []
    finished: list = []

    async def slow_tool(args, ctx):
        started.append(args["id"])
        await asyncio.sleep(0.05)
        finished.append(args["id"])
        return ToolResult(tool_name="slow", ok=True, output=args["id"])

    harness = Harness(
        tools=[_async_tool("slow", slow_tool)],
        validator=lambda state: ValidationResult(
            done=len(_tool_result_blocks(state)) >= 3,
            summary="all done",
        ),
    )
    batch = Decision.call_tools([
        ToolUseBlock(id="ta", name="slow", input={"id": "a"}),
        ToolUseBlock(id="tb", name="slow", input={"id": "b"}),
        ToolUseBlock(id="tc", name="slow", input={"id": "c"}),
    ])
    # v0.2-α: model emits FINAL_ANSWER after the parallel batch so the
    # validator gate at FINAL_ANSWER can finalize the run.
    runtime = _runtime(ScriptedModel(batch, Decision.final_answer("all done")))
    result = await runtime.arun(harness, "parallel test")

    assert result.status == RunStatus.COMPLETED
    assert sorted(started) == ["a", "b", "c"]
    blocks = _tool_result_blocks(result.state)
    assert len(blocks) == 3
    assert sorted(b.tool_use_id for b in blocks) == ["ta", "tb", "tc"]


# ---------------------------------------------------------------------------
# 10. Unknown tool
# ---------------------------------------------------------------------------


async def test_unknown_tool_yields_failed_observation():
    """Calling a nonexistent tool yields ToolResultBlock(is_error=True)."""
    runtime = _runtime(ScriptedModel(
        _call_tool("nope"),
        Decision.final_answer("acknowledged"),
    ))
    result = await runtime.arun(Harness(), "missing tool")
    assert result.status == RunStatus.COMPLETED

    blocks = _tool_result_blocks(result.state)
    assert len(blocks) >= 1
    last = blocks[-1]
    assert last.is_error is True
    text = last.content if isinstance(last.content, str) else json.dumps(
        [b.__dict__ for b in last.content], default=str
    )
    assert "unknown tool" in text


# ---------------------------------------------------------------------------
# 11. State serialization (v0.2 schema)
# ---------------------------------------------------------------------------


async def test_run_state_is_json_serializable():
    """RunState.to_dict serializes the v0.2 schema (messages, no goal/observations/feedback)."""
    runtime = _runtime(ScriptedModel(
        _call_tool("noop", {"x": 1}),
        Decision.final_answer("done"),
    ))

    def noop(args, ctx):
        return ToolResult(tool_name="noop", ok=True, output={"echoed": args})

    harness = Harness(
        tools=[_sync_tool("noop", noop)],
        validator=lambda state: ValidationResult(),
    )
    result = await runtime.arun(harness, "serialize me")
    snapshot = result.state.to_dict()

    # v0.2 shape: messages present; v0.1 keys gone.
    assert snapshot["schema_version"] == 2
    assert "messages" in snapshot
    assert "goal" not in snapshot
    assert "observations" not in snapshot
    assert "feedback" not in snapshot
    assert len(snapshot["messages"]) >= 1

    # First user message preserves the original goal text.
    first = snapshot["messages"][0]
    assert first["role"] == Role.USER.value
    assert any(
        b.get("type") == "text" and "serialize me" in b.get("text", "")
        for b in first["content"]
    )

    # Round-trip through JSON to prove durability.
    round_tripped = json.loads(json.dumps(snapshot, default=str))
    assert round_tripped["schema_version"] == 2
    assert round_tripped["iteration"] == result.state.iteration
    restored = RunState.from_dict(round_tripped)
    assert len(restored.messages) == len(result.state.messages)


# ---------------------------------------------------------------------------
# 12. Backward-compat: ToolResult positional args
# ---------------------------------------------------------------------------


def test_tool_result_positional_args_preserve_legacy_order():
    """ToolResult(name, ok, output, error) positional contract unchanged in v0.2."""
    r = ToolResult("echo", False, None, "boom")
    assert r.tool_name == "echo"
    assert r.ok is False
    assert r.error == "boom"
    assert r.attempts == 1
    assert r.call_id == ""
    assert r.duration_ms == 0.0


# ---------------------------------------------------------------------------
# 13. Decision string-kind compat
# ---------------------------------------------------------------------------


def test_decision_accepts_string_kind():
    """Decision(kind='final_answer', ...) — string-kind coercion still works."""
    d = Decision(
        kind="final_answer",  # type: ignore[arg-type]
        content=(TextBlock(text="hello"),),
    )
    assert d.kind == DecisionKind.FINAL_ANSWER
    assert d.kind.value == "final_answer"
    # v0.2: .text accessor replaces .content (str).
    assert d.text == "hello"


# ---------------------------------------------------------------------------
# 14. Tool timeout enforcement
# ---------------------------------------------------------------------------


async def test_tool_timeout_is_enforced():
    """Tool exceeding timeout_s yields ToolResultBlock(is_error=True)."""
    import asyncio

    async def slow(args, ctx):
        await asyncio.sleep(0.2)
        return ToolResult(tool_name="slow", ok=True, output="late")

    harness = Harness(
        tools=[Tool.from_async(
            name="slow",
            description="slow",
            handler=slow,
            timeout_s=0.05,
        )],
        validator=lambda state: ValidationResult(
            done=any(b.is_error for b in _tool_result_blocks(state)),
            summary="timed out as expected",
        ),
    )
    # v0.2-α: model emits FINAL_ANSWER after the timeout result so the
    # FINAL_ANSWER validator gate can complete the run.
    runtime = _runtime(ScriptedModel(
        _call_tool("slow"),
        Decision.final_answer("timed out as expected"),
    ))
    result = await runtime.arun(harness, "timeout test")
    assert result.status == RunStatus.COMPLETED

    blocks = _tool_result_blocks(result.state)
    assert blocks, "expected at least one tool result"
    assert blocks[-1].is_error is True


# ---------------------------------------------------------------------------
# 15. Retry policy
# ---------------------------------------------------------------------------


async def test_retry_policy_retries_then_succeeds():
    """RetryPolicy max_attempts=3 → flaky tool eventually succeeds, ToolResultBlock(is_error=False)."""
    attempts: list = []

    def flaky(args, ctx):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("not yet")
        return ToolResult(tool_name="flaky", ok=True, output="finally")

    harness = Harness(
        tools=[Tool.from_sync(
            name="flaky",
            description="retry me",
            handler=flaky,
            retry=RetryPolicy(max_attempts=3, backoff_initial_s=0.0),
        )],
        validator=lambda state: ValidationResult(
            done=any(not b.is_error for b in _tool_result_blocks(state)),
            summary="recovered via retry",
        ),
    )
    # v0.2-α: model emits FINAL_ANSWER after the retried tool succeeds so
    # the FINAL_ANSWER validator gate can complete the run.
    runtime = _runtime(ScriptedModel(
        _call_tool("flaky"),
        Decision.final_answer("recovered via retry"),
    ))
    result = await runtime.arun(harness, "retry test")
    assert result.status == RunStatus.COMPLETED
    assert result.summary == "recovered via retry"
    assert len(attempts) == 3

    blocks = _tool_result_blocks(result.state)
    assert blocks, "expected at least one tool result"
    assert blocks[-1].is_error is False


# ---------------------------------------------------------------------------
# 16. v0.1 schema rejection (extra coverage for migration boundary)
# ---------------------------------------------------------------------------


def test_runstate_rejects_v01_snapshot():
    """v0.1 RunState snapshots cannot load — UnknownSchemaVersionError."""
    v01_data = {
        "schema_version": 1,
        "goal": "old goal",
        "run_id": "x",
        "iteration": 0,
        "observations": [],
        "feedback": [],
        "trace": [],
        "interrupt": None,
        "metadata": {},
    }
    with pytest.raises(UnknownSchemaVersionError):
        RunState.from_dict(v01_data)
