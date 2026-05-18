"""v0.1 → v0.2 state dict migration (best-effort).

v0.1 ``RunState`` had: ``goal: str``, ``observations: list``,
``feedback: list``, ``schema_version: 1``.
v0.2 ``RunState`` has: ``messages: list[Message]``, ``schema_version: 2``.

This module reconstructs a v0.2 messages list from a v0.1 snapshot dict:

- ``goal`` → leading USER message with TextBlock.
- Each observation is migrated based on its ``kind``:
    - ``kind=="text"`` → a single ASSISTANT message with a TextBlock (the
      observation was free-form reasoning, not a tool call).
    - ``kind=="tool"`` or default (legacy snapshots without ``kind``) →
      an ASSISTANT(ToolUseBlock) + USER(ToolResultBlock) pair. Tool
      fields are read from either the top-level (``tool_call``, ``result``)
      OR a nested ``payload`` dict (``payload.call_id``, ``payload.output``,
      ...).
    - Unrecognized ``kind`` → an ASSISTANT TextBlock containing the
      payload's string representation (avoids producing a phantom tool
      call).
- Each ``feedback`` string → a USER TextBlock.
- Duplicate ``tool_use`` ids across observations are disambiguated with a
  suffix while preserving the use→result mapping.

This is best-effort and lossy: observation timing, exact tool IDs, and
ordering relative to feedback are reconstructed heuristically. For
production migrations, prefer re-running runs from scratch rather than
relying on this helper.

Codex P2 fixes (v0.2-α):
- :48: branch on ``obs.kind``; handle text observations; read tool fields
  from payload as well as top-level; do not fabricate tool calls for
  non-tool observations.
- :52: track seen tool_use_ids and regenerate duplicates while preserving
  the use→result mapping.
"""
from __future__ import annotations

from typing import Any
import uuid


def migrate_state_dict(v01_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a v0.1 RunState dict to a v0.2 RunState dict.

    Args:
        v01_dict: A RunState snapshot dict with schema_version == 1.

    Returns:
        A v0.2 RunState snapshot dict (schema_version == 2) with `messages`
        reconstructed.

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

    # Track every tool_use_id we've emitted so duplicates get disambiguated
    # while the original ids in the same observation's result still resolve
    # to the new id.
    seen_ids: set[str] = set()

    def _dedupe_id(raw: str) -> str:
        """Return a unique tool_use id, suffixing duplicates."""
        if raw and raw not in seen_ids:
            seen_ids.add(raw)
            return raw
        # Either the id is empty or already used. Generate a deterministic
        # disambiguated form, but make sure THAT isn't already used either.
        attempt = 0
        while True:
            attempt += 1
            candidate = (
                f"{raw}__dup{attempt}" if raw else f"call_{uuid.uuid4().hex[:8]}"
            )
            if candidate not in seen_ids:
                seen_ids.add(candidate)
                return candidate

    for obs in v01_dict.get("observations", []) or []:
        kind = obs.get("kind")
        payload = obs.get("payload")
        # Some v0.1 snapshots stored fields under payload; normalize.
        payload_dict = payload if isinstance(payload, dict) else {}

        if kind == "text":
            # Free-form reasoning observation — assistant TextBlock only.
            text = payload if isinstance(payload, str) else str(payload or "")
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            })
            continue

        # Tool observation (default for legacy snapshots without ``kind``).
        is_tool_observation = (
            kind == "tool"
            or kind is None  # legacy: no kind field, treat as tool obs
            or "tool_call" in obs
            or "tool_name" in obs
            or "call_id" in payload_dict
        )

        if not is_tool_observation:
            # Unknown kind — degrade gracefully into an assistant TextBlock
            # rather than fabricating a tool call. Preserves the payload as
            # readable text so the data isn't silently lost.
            text = (
                payload if isinstance(payload, str)
                else str(payload) if payload is not None
                else ""
            )
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            })
            continue

        # Resolve tool fields from either the top-level or the payload dict.
        tool_call = obs.get("tool_call") or {}
        tool_name = (
            tool_call.get("name")
            or obs.get("tool_name")
            or payload_dict.get("name")
            or "unknown"
        )
        tool_args = (
            tool_call.get("input")
            or obs.get("input")
            or payload_dict.get("input")
            or {}
        )
        raw_id = (
            tool_call.get("id")
            or obs.get("id")
            or payload_dict.get("call_id")
            or payload_dict.get("id")
            or f"call_{uuid.uuid4().hex[:8]}"
        )
        tool_use_id = _dedupe_id(raw_id)

        messages.append({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": tool_args,
            }],
        })

        result_content = (
            obs.get("result")
            if obs.get("result") is not None
            else obs.get("output")
            if obs.get("output") is not None
            else payload_dict.get("output")
            if payload_dict.get("output") is not None
            else payload_dict.get("result")
            if payload_dict.get("result") is not None
            else ""
        )
        if not isinstance(result_content, str):
            result_content = str(result_content)
        is_error = bool(
            obs.get("error")
            or obs.get("is_error")
            or payload_dict.get("error")
            or payload_dict.get("is_error")
        )

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
