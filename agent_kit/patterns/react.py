"""ReAct pattern: a Harness pre-configured for tool-using agents.

This is the most basic pattern: the model sees observations, decides whether
to call a tool or emit a final answer, and the loop iterates. It is the
default starting point for any agent built on agent_core.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from agent_core import (
    Decision,
    GuardResult,
    Harness,
    RunState,
    Tool,
    ValidationResult,
)


DEFAULT_REACT_SYSTEM_PROMPT = """You are an AI agent solving a user's goal step by step.

You have access to tools. On every turn you either:
1. Call one or more tools to gather information or take action, OR
2. Produce a final natural-language answer when the goal is satisfied.

Guidelines:
- Prefer making concrete progress over deliberating.
- When a tool returns an error, read the error and try a different approach.
- Stop and produce a final answer as soon as the goal is met.
- Be concise in your final answer."""


def react_harness(
    tools: Sequence[Tool],
    *,
    system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT,
    max_observations_in_context: Optional[int] = None,
    guard=None,
    validator=None,
) -> Harness:
    """Build a Harness pre-wired for ReAct-style agent loops.

    Parameters
    ----------
    tools:
        Tools the agent may invoke.
    system_prompt:
        Overrides the default ReAct system prompt.
    max_observations_in_context:
        If set, only the most recent N observation payloads are included in
        the per-turn context (cheap windowing; for token-aware truncation
        use a real Compactor).
    guard:
        Custom guard fn ``(decision, state) -> GuardResult``.
    validator:
        Custom validator fn ``(state) -> ValidationResult``. If omitted the
        loop relies solely on the model's ``final_answer`` decision.
    """

    def _context_builder(state: RunState, _tools: Sequence[Tool]) -> Mapping[str, object]:
        obs_payloads = [obs.payload for obs in state.observations if obs.kind == "tool_result"]
        if max_observations_in_context is not None:
            obs_payloads = obs_payloads[-max_observations_in_context:]
        return {
            "system_prompt": system_prompt,
            "goal": state.goal,
            "iteration": state.iteration,
            "observations": obs_payloads,
            "feedback": list(state.feedback),
        }

    return Harness(
        tools=list(tools),
        guard=guard or (lambda decision, state: GuardResult()),
        validator=validator or (lambda state: ValidationResult()),
        context_builder=_context_builder,
    )
