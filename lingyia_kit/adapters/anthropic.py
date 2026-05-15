"""Anthropic (Claude) Model adapter.

Claude's ``messages`` API uses a different shape from OpenAI:
- ``tool_use`` blocks inside ``content`` instead of top-level ``tool_calls``
- ``tool_result`` blocks instead of ``role="tool"`` messages
- Mandatory system prompt as a separate ``system`` parameter, not a message

Uses the ``anthropic`` Python SDK (already in this environment).
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from lingyia_core import Decision, RunState, ToolCall
from lingyia_core.state import ModelUsage

try:
    from anthropic import AsyncAnthropic
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The anthropic SDK is required for AnthropicModel. "
        "Install it with `pip install anthropic`."
    ) from exc


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI agent. Use the available tools to accomplish the user's goal. "
    "When the goal is fully accomplished, respond with a final text answer instead of calling a tool."
)


class AnthropicModel:
    """Claude messages-API adapter implementing the Model protocol."""

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        client: Optional[AsyncAnthropic] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client = client or AsyncAnthropic(api_key=api_key, timeout=timeout_s)

    async def adecide(
        self,
        context: Mapping[str, Any],
        state: RunState,
        tools: Sequence[Any],
    ) -> Decision:
        system_prompt = context.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        messages = self._build_messages(state)
        tool_schemas = self._build_tool_schemas(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": messages,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    # Message construction -----------------------------------------------

    def _build_messages(self, state: RunState) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": state.goal},
        ]

        by_iter: dict[int, list[Any]] = {}
        for obs in state.observations:
            if obs.kind != "tool_result":
                continue
            by_iter.setdefault(obs.iteration, []).append(obs)

        for iter_num in sorted(by_iter.keys()):
            results = by_iter[iter_num]
            assistant_content = []
            for i, r in enumerate(results):
                payload = r.payload
                call_id = payload.get("call_id") or f"toolu_{iter_num}_{i}"
                assistant_content.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": payload["tool_name"],
                    "input": payload.get("tool_args", {}),
                })
            messages.append({"role": "assistant", "content": assistant_content})

            user_content = []
            for r in results:
                payload = r.payload
                call_id = payload.get("call_id") or ""
                result_content = json.dumps(
                    {
                        "ok": payload.get("ok", True),
                        "output": payload.get("output"),
                        "error": payload.get("error", ""),
                    },
                    default=str,
                    ensure_ascii=False,
                )
                user_content.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": result_content,
                    "is_error": not payload.get("ok", True),
                })
            messages.append({"role": "user", "content": user_content})

        for fb in state.feedback:
            messages.append({"role": "user", "content": f"[feedback] {fb}"})

        return messages

    def _build_tool_schemas(self, tools: Sequence[Any]) -> list[dict[str, Any]]:
        schemas = []
        for t in tools:
            input_schema = dict(t.input_schema) if t.input_schema else {
                "type": "object",
                "properties": {},
            }
            schemas.append({
                "name": t.name,
                "description": t.description,
                "input_schema": input_schema,
            })
        return schemas

    # Response parsing ---------------------------------------------------

    def _parse_response(self, response: Any) -> Decision:
        blocks = getattr(response, "content", None) or []
        tool_calls = []
        text_parts = []
        for block in blocks:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                tool_calls.append(ToolCall(
                    name=getattr(block, "name", ""),
                    args=dict(getattr(block, "input", {}) or {}),
                    call_id=getattr(block, "id", "") or "",
                ))
            elif btype == "text":
                text_parts.append(getattr(block, "text", "") or "")

        usage = self._parse_usage(getattr(response, "usage", None))

        if tool_calls:
            return Decision(
                kind=Decision.call_tools(tuple(tool_calls)).kind,
                tool_calls=tuple(tool_calls),
                usage=usage,
            )
        content = "\n".join(text_parts).strip()
        return Decision(
            kind=Decision.final_answer(content).kind,
            content=content,
            usage=usage,
        )

    def _parse_usage(self, u: Any) -> ModelUsage:
        if u is None:
            return ModelUsage(model_id=self.model)
        # Anthropic SDK returns a typed object; fall back to dict-style for tests.
        def _get(name: str, default: int = 0) -> int:
            return int(getattr(u, name, None) or (u.get(name) if isinstance(u, dict) else None) or default)

        input_tokens = _get("input_tokens")
        output_tokens = _get("output_tokens")
        cache_read = _get("cache_read_input_tokens")
        cache_create = _get("cache_creation_input_tokens")
        # We track cache_read as "cached_tokens" since they're discounted.
        # cache_creation tokens are billed at the regular prompt rate, so we
        # leave them inside prompt_tokens.
        return ModelUsage(
            prompt_tokens=input_tokens + cache_create + cache_read,
            completion_tokens=output_tokens,
            cached_tokens=cache_read,
            model_id=self.model,
        )
