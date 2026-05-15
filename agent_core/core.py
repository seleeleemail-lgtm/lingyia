from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


ToolHandler = Callable[[Dict[str, Any], "RunState"], "ToolResult"]
GuardFn = Callable[["Decision", "RunState"], "GuardResult"]
ContextBuilder = Callable[["RunState", "Harness"], Dict[str, Any]]
Validator = Callable[["RunState"], "ValidationResult"]


@dataclass(frozen=True)
class Decision:
    kind: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    content: str = ""

    @classmethod
    def call_tool(cls, tool_name: str, tool_args: Optional[Dict[str, Any]] = None) -> "Decision":
        return cls(kind="call_tool", tool_name=tool_name, tool_args=tool_args or {})

    @classmethod
    def final_answer(cls, content: str) -> "Decision":
        return cls(kind="final_answer", content=content)

    @classmethod
    def ask_human(cls, question: str) -> "Decision":
        return cls(kind="ask_human", content=question)

    @classmethod
    def abort(cls, reason: str) -> "Decision":
        return cls(kind="abort", content=reason)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: Any = None
    error: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: ToolHandler

    def run(self, args: Dict[str, Any], state: "RunState") -> ToolResult:
        return self.handler(args, state)


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


@dataclass
class RunState:
    goal: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    iteration: int = 0
    observations: List[Any] = field(default_factory=list)
    tool_history: List[ToolResult] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    pending_decision: Optional[Decision] = None

    def add_feedback(self, feedback: str) -> None:
        if feedback:
            self.feedback.append(feedback)


@dataclass(frozen=True)
class RunResult:
    status: str
    state: RunState
    summary: str = ""
    reason: str = ""


def _default_context_builder(state: RunState, harness: "Harness") -> Dict[str, Any]:
    return {
        "goal": state.goal,
        "iteration": state.iteration,
        "observations": list(state.observations),
        "feedback": list(state.feedback),
        "tools": [
            {"name": tool.name, "description": tool.description}
            for tool in harness.tools
        ],
    }


def _default_guard(decision: Decision, state: RunState) -> GuardResult:
    return GuardResult()


def _default_validator(state: RunState) -> ValidationResult:
    return ValidationResult()


@dataclass
class Harness:
    tools: List[Tool] = field(default_factory=list)
    validate_progress: Validator = _default_validator
    guard_action: GuardFn = _default_guard
    build_context: ContextBuilder = _default_context_builder
    max_iterations: int = 8

    def tool_by_name(self, name: str) -> Optional[Tool]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


class AgentLoop:
    def __init__(self, model: Any, harness: Harness):
        self.model = model
        self.harness = harness

    def run(self, goal: str) -> RunResult:
        state = RunState(goal=goal)
        return self._continue(state)

    def resume(self, state: RunState, approved: bool, feedback: str = "") -> RunResult:
        if state.pending_decision is None:
            return self._continue(state)

        decision = state.pending_decision
        state.pending_decision = None
        if not approved:
            state.add_feedback(feedback or "approval rejected")
            state.iteration += 1
            return self._continue(state)

        result = self._execute_tool_decision(state, decision)
        if result is not None:
            return result
        return self._continue(state)

    def _continue(self, state: RunState) -> RunResult:
        while state.iteration < self.harness.max_iterations:
            context = self.harness.build_context(state, self.harness)
            decision = self.model.decide(context, state)
            state.trace.append(
                {
                    "iteration": state.iteration,
                    "event": "decision",
                    "kind": decision.kind,
                    "tool_name": decision.tool_name,
                }
            )

            if decision.kind == "final_answer":
                verdict = self.harness.validate_progress(state)
                if verdict.needs_human:
                    return RunResult(status="paused", state=state, reason=verdict.question)
                return RunResult(status="completed", state=state, summary=verdict.summary or decision.content)

            if decision.kind == "ask_human":
                return RunResult(status="paused", state=state, reason=decision.content)

            if decision.kind == "abort":
                return RunResult(status="failed", state=state, reason=decision.content)

            if decision.kind != "call_tool":
                return RunResult(status="failed", state=state, reason="unknown decision kind: %s" % decision.kind)

            guard = self.harness.guard_action(decision, state)
            if guard.requires_approval:
                state.pending_decision = decision
                return RunResult(status="approval_required", state=state, reason=guard.reason)
            if not guard.allowed:
                state.add_feedback(guard.reason)
                state.iteration += 1
                continue

            result = self._execute_tool_decision(state, decision)
            if result is not None:
                return result

        return RunResult(status="stopped", state=state, reason="max_iterations reached")

    def _execute_tool_decision(self, state: RunState, decision: Decision) -> Optional[RunResult]:
        if not decision.tool_name:
            return RunResult(status="failed", state=state, reason="tool decision missing tool name")

        tool = self.harness.tool_by_name(decision.tool_name)
        if tool is None:
            return RunResult(status="failed", state=state, reason="unknown tool: %s" % decision.tool_name)

        try:
            result = tool.run(decision.tool_args, state)
        except Exception as exc:
            result = ToolResult(tool_name=tool.name, ok=False, error=str(exc))

        state.tool_history.append(result)
        state.observations.append(result)
        state.trace.append(
            {
                "iteration": state.iteration,
                "event": "tool_result",
                "tool_name": result.tool_name,
                "ok": result.ok,
            }
        )

        verdict = self.harness.validate_progress(state)
        if verdict.done:
            return RunResult(status="completed", state=state, summary=verdict.summary)
        if verdict.needs_human:
            return RunResult(status="paused", state=state, reason=verdict.question)

        state.add_feedback(verdict.feedback)
        state.iteration += 1
        return None
