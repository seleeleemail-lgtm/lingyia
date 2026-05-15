"""OpenAI-compatible Model adapter base.

Targets the ``/v1/chat/completions`` endpoint with OpenAI-style tool-calling.
Works for OpenAI itself, SiliconFlow, MiniMax, OpenRouter, vLLM, llama.cpp,
LM Studio, and any other provider that conforms to the same shape.

Implementation notes:
- Pure ``httpx`` client. We deliberately do not depend on the ``openai`` SDK,
  so the only required wire-level dependency is HTTP+JSON.
- Conversation history is reconstructed from ``RunState.observations`` on
  every turn. Each ``tool_result`` observation carries ``tool_args``, which
  lets us rebuild the matching assistant ``tool_calls`` message.
- Parallel tool calls in a single iteration are collapsed into one assistant
  turn followed by N tool messages, matching the OpenAI protocol.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

import httpx

from lingyia_core import Decision, RunState, ToolCall
from lingyia_core.state import ModelUsage


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI agent. Use the available tools to accomplish the user's goal. "
    "When the goal is fully accomplished, respond with a final answer instead of calling a tool."
)


class OpenAICompatibleModel:
    """Base adapter for any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def adecide(
        self,
        context: Mapping[str, Any],
        state: RunState,
        tools: Sequence[Any],
    ) -> Decision:
        messages = self._build_messages(context, state)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        tool_schemas = self._build_tool_schemas(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
        # Caller-supplied overrides win (e.g. response_format, top_p, ...).
        payload.update(self.extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    # Message construction -----------------------------------------------

    def _build_messages(
        self,
        context: Mapping[str, Any],
        state: RunState,
    ) -> list[dict[str, Any]]:
        system_prompt = context.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state.goal},
        ]

        # Group tool_result observations by iteration so each iteration becomes
        # exactly one assistant turn + N tool result messages.
        by_iter: dict[int, list[Any]] = {}
        for obs in state.observations:
            if obs.kind != "tool_result":
                continue
            by_iter.setdefault(obs.iteration, []).append(obs)

        for iter_num in sorted(by_iter.keys()):
            results = by_iter[iter_num]
            tool_calls_payload = []
            for i, r in enumerate(results):
                payload = r.payload
                call_id = payload.get("call_id") or f"call_{iter_num}_{i}"
                tool_calls_payload.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": payload["tool_name"],
                        "arguments": json.dumps(payload.get("tool_args", {}), ensure_ascii=False),
                    },
                })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_payload,
            })
            for r in results:
                payload = r.payload
                call_id = payload.get("call_id") or ""
                content = json.dumps(
                    {
                        "ok": payload.get("ok", True),
                        "output": payload.get("output"),
                        "error": payload.get("error", ""),
                    },
                    default=str,
                    ensure_ascii=False,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                })

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
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": input_schema,
                },
            })
        return schemas

    # Response parsing ---------------------------------------------------

    def _parse_response(self, body: Mapping[str, Any]) -> Decision:
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"empty choices in response: {body}")
        msg = choices[0].get("message") or {}
        usage = self._parse_usage(body)

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            calls = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        # Some smaller models emit malformed JSON; degrade
                        # gracefully so the loop sees a real failure instead
                        # of crashing.
                        args = {"_raw": raw_args}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                calls.append(ToolCall(name=name, args=args, call_id=tc.get("id") or ""))
            if calls:
                return Decision(
                    kind=Decision.call_tools(tuple(calls)).kind,
                    tool_calls=tuple(calls),
                    usage=usage,
                )

        content = msg.get("content") or ""
        return Decision(
            kind=Decision.final_answer(content).kind,
            content=content,
            usage=usage,
        )

    def _parse_usage(self, body: Mapping[str, Any]) -> ModelUsage:
        u = body.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        return ModelUsage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            cached_tokens=cached,
            model_id=self.model,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
