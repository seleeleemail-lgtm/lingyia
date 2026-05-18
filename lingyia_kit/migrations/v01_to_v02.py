"""v0.1 → v0.2 state dict migration (best-effort).

v0.1 RunState had: goal: str, observations: list, feedback: list, schema_version: 1
v0.2 RunState has: messages: list[Message], schema_version: 2

This module reconstructs a v0.2 messages list from a v0.1 snapshot dict:
- goal → USER message with TextBlock
- Each observation → ASSISTANT (ToolUseBlock) + USER (ToolResultBlock) pair
- Each feedback string → USER TextBlock

This is best-effort and lossy: observation timing, tool IDs, and ordering relative
to feedback are reconstructed heuristically. For production migrations, prefer
re-running runs from scratch rather than relying on this helper.
"""
from __future__ import annotations

from typing import Any
import uuid


def migrate_state_dict(v01_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a v0.1 RunState dict to a v0.2 RunState dict.

    Args:
        v01_dict: A RunState snapshot dict with schema_version == 1.

    Returns:
        A v0.2 RunState snapshot dict (schema_version == 2) with `messages` reconstructed.

    Raises:
        ValueError: If the input is not a v0.1 snapshot.
    """
    if v01_dict.get("schema_version") != 1:
        raise ValueError(
            f"migrate_state_dict expects schema_version == 1, "
            f"got {v01_dict.get('schema_version')!r}"
        )

    messages: list[dict[str, Any]] = []

    goal = v01_dict.get("goal", "")
    if goal:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": goal}],
        })

    for obs in v01_dict.get("observations", []) or []:
        tool_call = obs.get("tool_call") or {}
        tool_name = tool_call.get("name") or obs.get("tool_name") or "unknown"
        tool_args = tool_call.get("input") or obs.get("input") or {}
        tool_use_id = tool_call.get("id") or obs.get("id") or f"call_{uuid.uuid4().hex[:8]}"

        messages.append({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": tool_args,
            }],
        })

        result_content = obs.get("result") or obs.get("output") or ""
        if not isinstance(result_content, str):
            result_content = str(result_content)
        is_error = bool(obs.get("error") or obs.get("is_error"))

        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_content,
                "is_error": is_error,
            }],
        })

    for fb in v01_dict.get("feedback", []) or []:
        if not fb:
            continue
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": str(fb)}],
        })

    v02_dict: dict[str, Any] = {
        "messages": messages,
        "run_id": v01_dict.get("run_id", f"run_{uuid.uuid4().hex[:8]}"),
        "iteration": v01_dict.get("iteration", 0),
        "trace": v01_dict.get("trace", []),
        "interrupt": v01_dict.get("interrupt"),
        "metadata": dict(v01_dict.get("metadata", {})),
        "schema_version": 2,
    }
    return v02_dict
