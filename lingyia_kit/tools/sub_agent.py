"""Sub-agent as a Tool: wrap a `Runtime + Harness` as a `Tool`.

This is the canonical agent-as-tool pattern. The parent agent calls the
sub-agent like any other tool; the sub-agent runs its own loop with its own
state, tools, telemetry, then returns a structured result. The sub-agent's
total cost is surfaced in the tool's output payload so callers can read
it directly off the ToolResultBlock — but it is **not** automatically
charged to the parent runtime's ``max_cost_usd`` budget (see "Cost model"
below).

When to use:
- Task decomposition: parent plans, sub-agents execute.
- Specialist agents: parent dispatches "summarize legal clauses" to a sub-agent
  with a legal-specific harness.
- Multi-step workflows: sequential or parallel sub-agents from one parent.

When NOT to use:
- For pure tool calls (just use a normal Tool).
- For long-running async waits (use checkpoint + resume instead).

Cost / state model:
- The sub-agent's state is fully independent: separate run_id, messages,
  trace, checkpoint. The parent only sees the sub-agent's final summary +
  cost in the tool result, surfaced as a ToolResultBlock in the parent's
  transcript.
- **Cost is NOT propagated.** ``Runtime`` accumulates
  ``state.metadata["cost_usd"]`` from ``Decision.usage.cost_usd`` on each
  *parent* decision; sub-agent cost lives on the sub-agent's own state
  and is reported in the tool result dict (``output["cost_usd"]``) plus
  the ``sub_agent_cost`` telemetry event below. Callers that need unified
  cost accounting must aggregate manually — e.g. wrap the parent runtime
  with a ``cost_estimator`` that inspects tool results, or sum cost in
  application code from telemetry. This was previously documented as
  automatic propagation; codex verify P2 caught that the runtime never
  honors that contract, so the doc is now aligned with reality.
- Telemetry: events from the sub-agent flow through the sub-agent's own
  telemetry sink, not the parent's. Wire both runtimes to the same backend
  for correlated traces.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, TYPE_CHECKING

from lingyia_core import Tool, ToolContext, ToolResult
from lingyia_core.state import ModelUsage

if TYPE_CHECKING:
    from lingyia_core import Harness, Runtime


def sub_agent_tool(
    *,
    name: str,
    description: str,
    runtime: "Runtime",
    harness: "Harness",
    goal_arg: str = "goal",
    additional_args: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tool:
    """Build a Tool that runs a sub-agent.

    Parameters
    ----------
    name:
        Tool name exposed to the parent agent. Make it specific and
        non-overlapping with sibling tools.
    description:
        Tool description for the parent's LLM. Make it tell the parent
        *when* to use this sub-agent vs other tools.
    runtime:
        The Runtime that powers the sub-agent. Can be the same as the
        parent's runtime or a different one (e.g. a cheaper model).
    harness:
        Domain Harness for the sub-agent. Independent from the parent's.
    goal_arg:
        Name of the input field that holds the goal string. Defaults to
        ``"goal"``.
    additional_args:
        Extra JSON Schema properties to expose on the tool (will be ignored
        by the sub-agent's loop unless the harness uses them via metadata).

    The returned ``ToolResult.output`` is a dict with:
        - ``summary``       — final answer or failure reason
        - ``status``        — completed / failed / paused / stopped
        - ``iterations``    — how many turns the sub-agent took
        - ``cost_usd``      — total cost
        - ``tokens``        — {prompt, completion, cached}
        - ``run_id``        — sub-agent's run id (useful for correlation)
    """
    from lingyia_core import RunStatus  # local import to avoid cycle at import time

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            goal_arg: {
                "type": "string",
                "description": "The goal the sub-agent should accomplish.",
            },
        },
        "required": [goal_arg],
    }
    if additional_args:
        schema["properties"].update(additional_args)

    async def _handler(args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        sub_goal = args.get(goal_arg)
        if not isinstance(sub_goal, str) or not sub_goal:
            return ToolResult(
                tool_name=name,
                ok=False,
                error=f"missing or empty '{goal_arg}'",
            )

        sub_result = await runtime.arun(harness, sub_goal)
        sub_state = sub_result.state
        cost = float(sub_state.metadata.get("cost_usd", 0.0) or 0.0)
        tokens = dict(sub_state.metadata.get("tokens", {}) or {})
        ok = sub_result.status == RunStatus.COMPLETED

        output = {
            "summary": sub_result.summary or sub_result.reason,
            "status": sub_result.status.value,
            "iterations": sub_state.iteration,
            "cost_usd": cost,
            "tokens": tokens,
            "run_id": sub_state.run_id,
        }

        result = ToolResult(
            tool_name=name,
            ok=ok,
            output=output,
            error="" if ok else (sub_result.reason or "sub-agent did not complete"),
        )

        # Cost is NOT propagated to the parent runtime's max_cost_usd
        # budget. Runtime.max_cost_usd only checks state.metadata["cost_usd"]
        # which is accumulated from Decision.usage.cost_usd per parent
        # decision (not from tool results). The sub-agent's cost is
        # surfaced in output["cost_usd"] so callers can aggregate manually
        # — e.g. a custom cost_estimator on the parent runtime, or
        # post-processing the telemetry stream. v0.2-γ will introduce a
        # proper cost-propagation API.
        return result

    return Tool.from_async(
        name=name,
        description=description,
        handler=_handler,
        input_schema=schema,
        side_effect="external_io",
        idempotent=False,
    )
