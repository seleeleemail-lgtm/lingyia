"""ReAct pattern factory: a Harness with sensible defaults.

v0.2: no longer builds context_builder with observations/feedback. Messages
are on state directly; adapters read them. Harness just packages tools +
system_prompt + metadata.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from lingyia_core import Harness
from lingyia_core.executor import Tool


DEFAULT_REACT_SYSTEM_PROMPT = (
    "You are a helpful AI agent. Use the available tools to accomplish the user's request. "
    "When done, respond with a final answer."
)


def react_harness(
    *,
    tools: Sequence[Tool] = (),
    system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT,
    metadata: Optional[Mapping[str, Any]] = None,
    granted_permissions: Optional[frozenset[str]] = None,
) -> Harness:
    """Build a ReAct-style harness. v0.2: simplified."""
    kwargs: dict[str, Any] = {
        "tools": list(tools),
        "system_prompt": system_prompt,
    }
    if metadata is not None:
        kwargs["metadata"] = dict(metadata)
    if granted_permissions is not None:
        kwargs["granted_permissions"] = granted_permissions
    return Harness(**kwargs)
